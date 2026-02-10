
"""
Tabula-Rasa 3D Physics Discovery (PyBullet + Gymnasium + PPO + RND + World-Model logging)

- 3D rigid-body physics world (PyBullet)
- Point agent (sphere) with minimal actions (forward force + yaw torque)
- No pretrained weights: PPO policy, RND predictor, and world model start from random init
- Intrinsic reward: Random Network Distillation (RND) novelty bonus (Burda et al., 2018)
- Diagnostics: world-model prediction loss, contact events, object displacement, state-space coverage
- Logging: TensorBoard scalars + text events + optional videos

References:
- PyBullet quickstart/overview: https://github.com/bulletphysics/bullet3/tree/master/docs
- Gymnasium Env API: https://gymnasium.farama.org/api/env/
- SB3 PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- RND paper: https://arxiv.org/abs/1810.12894
- ICM paper (alternative curiosity): https://arxiv.org/abs/1705.05363
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List, Deque

import numpy as np

import gymnasium as gym
from gymnasium import spaces

# PyBullet
import pybullet as p
import pybullet_data

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# RL
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


# ---------------------------
# Utility: running mean/std
# ---------------------------

class RunningMeanStd:
    """
    Numerically stable running mean/std for normalization.
    """
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)


# ---------------------------
# Environment: PyBullet 3D world
# ---------------------------

@dataclass
class WorldConfig:
    world_size: float = 6.0          # half-width of square arena
    max_steps: int = 600
    time_step: float = 1.0 / 120.0
    frame_skip: int = 4              # physics steps per agent action
    gravity: float = -9.81

    # Sensing
    num_rays: int = 24
    ray_length: float = 8.0
    ray_height: float = 0.35

    # Agent
    agent_radius: float = 0.12
    agent_mass: float = 1.0
    max_forward_force: float = 8.0
    max_yaw_torque: float = 1.5

    # Objects
    base_num_objects: int = 4        # curriculum level 0
    max_objects: int = 20
    object_mass_range: Tuple[float, float] = (0.3, 4.0)
    object_size_range: Tuple[float, float] = (0.08, 0.25)

    # Curriculum toggles
    enable_ramp_at_level: int = 2
    enable_second_room_at_level: int = 3


class BulletWorldEnv(gym.Env):
    """
    A minimal 3D rigid-body world with:
    - bounded arena (walls)
    - point agent (sphere) that moves via forces/torques
    - a set of dynamic objects (boxes/spheres/cylinders)
    - optional ramp / second room (curriculum)

    Observation (vector, float32):
    [ agent_pos(3), agent_vel(3), yaw_sin, yaw_cos, ang_vel_z, rays(num_rays) ]
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        cfg: WorldConfig,
        render_mode: Optional[str] = None,
        gui: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.gui = gui
        self._seed = seed
        self._np_random = np.random.default_rng(seed)

        # Curriculum state
        self.curriculum_level = 0

        # PyBullet state
        self._client_id: Optional[int] = None
        self._agent_id: Optional[int] = None
        self._object_ids: List[int] = []
        self._step_count = 0

        # Track object positions to compute per-step displacement
        self._last_obj_pos: Dict[int, np.ndarray] = {}

        # Observation/action spaces
        obs_dim = 3 + 3 + 2 + 1 + self.cfg.num_rays  # pos, vel, yaw(sin/cos), ang_vel_z, rays
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action: [forward_throttle, yaw_throttle] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    # ---- Gym API ----

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._seed = seed
            self._np_random = np.random.default_rng(seed)

        if self._client_id is None:
            self._connect()

        p.resetSimulation(physicsClientId=self._client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client_id)
        p.setGravity(0, 0, self.cfg.gravity, physicsClientId=self._client_id)
        p.setTimeStep(self.cfg.time_step, physicsClientId=self._client_id)

        self._step_count = 0
        self._object_ids = []
        self._last_obj_pos = {}

        # Ground plane
        p.loadURDF("plane.urdf", physicsClientId=self._client_id)

        # Arena walls
        self._build_arena()

        # Curriculum: add structures
        if self.curriculum_level >= self.cfg.enable_ramp_at_level:
            self._build_ramp()
        if self.curriculum_level >= self.cfg.enable_second_room_at_level:
            self._build_second_room()

        # Agent
        self._agent_id = self._spawn_agent()

        # Objects
        num_objects = min(self.cfg.base_num_objects + 2 * self.curriculum_level, self.cfg.max_objects)
        for _ in range(num_objects):
            oid = self._spawn_random_object()
            self._object_ids.append(oid)
            self._last_obj_pos[oid] = np.array(p.getBasePositionAndOrientation(oid, physicsClientId=self._client_id)[0])

        obs = self._get_obs()
        info = self._get_info_base()
        return obs, info

    def step(self, action: np.ndarray):
        assert self._client_id is not None and self._agent_id is not None
        self._step_count += 1

        action = np.asarray(action, dtype=np.float32)
        forward = float(np.clip(action[0], -1, 1))
        yaw = float(np.clip(action[1], -1, 1))

        force_mag = forward * self.cfg.max_forward_force
        torque_mag = yaw * self.cfg.max_yaw_torque

        # Step physics with persistent control across substeps.
        # PyBullet clears external forces after each stepSimulation(),
        # so we must re-apply them every substep.
        for _ in range(self.cfg.frame_skip):
            # Read agent yaw from orientation
            pos, orn = p.getBasePositionAndOrientation(self._agent_id, physicsClientId=self._client_id)
            yaw_angle = p.getEulerFromQuaternion(orn)[2]

            # Apply force in current heading direction (world frame)
            fx = math.cos(yaw_angle) * force_mag
            fy = math.sin(yaw_angle) * force_mag
            p.applyExternalForce(
                objectUniqueId=self._agent_id,
                linkIndex=-1,
                forceObj=[fx, fy, 0.0],
                posObj=pos,
                flags=p.WORLD_FRAME,
                physicsClientId=self._client_id,
            )

            # Apply yaw torque (world frame)
            p.applyExternalTorque(
                objectUniqueId=self._agent_id,
                linkIndex=-1,
                torqueObj=[0.0, 0.0, torque_mag],
                flags=p.WORLD_FRAME,
                physicsClientId=self._client_id,
            )

            p.stepSimulation(physicsClientId=self._client_id)

        obs = self._get_obs()
        info = self._get_info_base()

        # No extrinsic reward in this setup (wrappers add intrinsic rewards)
        reward = 0.0

        # Termination/truncation
        terminated = False
        truncated = self._step_count >= self.cfg.max_steps

        # Basic safety termination: fell under plane or escaped arena (rare with walls)
        agent_pos = np.array(p.getBasePositionAndOrientation(self._agent_id, physicsClientId=self._client_id)[0])
        if agent_pos[2] < 0.05:
            terminated = True
            info["termination_reason"] = "fell"

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        assert self._client_id is not None and self._agent_id is not None

        # Simple chase camera
        pos, orn = p.getBasePositionAndOrientation(self._agent_id, physicsClientId=self._client_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        cam_dist = 2.2
        cam_height = 1.2
        target = [pos[0], pos[1], pos[2] + 0.25]
        cam_pos = [
            pos[0] - cam_dist * math.cos(yaw),
            pos[1] - cam_dist * math.sin(yaw),
            pos[2] + cam_height,
        ]

        view = p.computeViewMatrix(cameraEyePosition=cam_pos, cameraTargetPosition=target, cameraUpVector=[0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov=70, aspect=1.0, nearVal=0.05, farVal=50.0)
        w, h = 256, 256
        img = p.getCameraImage(
            width=w, height=h, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._client_id
        )
        rgb = np.reshape(img[2], (h, w, 4))[:, :, :3].astype(np.uint8)
        if self.render_mode == "rgb_array":
            return rgb
        elif self.render_mode == "human":
            # In GUI mode PyBullet shows window; nothing to do.
            return None
        return None

    def close(self):
        if self._client_id is not None:
            p.disconnect(physicsClientId=self._client_id)
            self._client_id = None

    # ---- Curriculum control ----

    def set_curriculum_level(self, level: int) -> None:
        self.curriculum_level = int(max(0, level))

    # ---- Internal helpers ----

    def _connect(self):
        if self.gui:
            self._client_id = p.connect(p.GUI)
        else:
            self._client_id = p.connect(p.DIRECT)

    def _build_arena(self):
        assert self._client_id is not None
        half = self.cfg.world_size
        wall_thick = 0.12
        wall_height = 0.6
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, wall_thick, wall_height], physicsClientId=self._client_id)
        # +Y wall
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            basePosition=[0, half, wall_height],
            physicsClientId=self._client_id,
        )
        # -Y wall
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, basePosition=[0, -half, wall_height], physicsClientId=self._client_id)

        col2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thick, half, wall_height], physicsClientId=self._client_id)
        # +X wall
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col2, basePosition=[half, 0, wall_height], physicsClientId=self._client_id)
        # -X wall
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col2, basePosition=[-half, 0, wall_height], physicsClientId=self._client_id)

        # Central occluder wall (teaches occlusion/object permanence)
        occ_len = half * 0.9
        occ_thick = 0.08
        occ_height = 0.8
        occ_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[occ_len * 0.5, occ_thick, occ_height], physicsClientId=self._client_id)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=occ_col, basePosition=[0, 0, occ_height], physicsClientId=self._client_id)

    def _build_ramp(self):
        assert self._client_id is not None
        # Simple tilted box ramp
        ramp_half = [1.2, 0.6, 0.05]
        ramp_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=ramp_half, physicsClientId=self._client_id)
        ramp_pos = [-(self.cfg.world_size * 0.35), -(self.cfg.world_size * 0.25), 0.1]
        ramp_yaw = 0.3
        ramp_pitch = -0.35
        orn = p.getQuaternionFromEuler([0.0, ramp_pitch, ramp_yaw])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ramp_col, basePosition=ramp_pos, baseOrientation=orn, physicsClientId=self._client_id)

    def _build_second_room(self):
        assert self._client_id is not None
        # Add a doorway wall splitting arena (creates partial observability/navigation)
        half = self.cfg.world_size
        wall_thick = 0.10
        wall_height = 0.6
        gap = 0.8  # doorway gap
        # left segment
        seg1_len = (half - gap * 0.5)
        col1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[seg1_len * 0.5, wall_thick, wall_height], physicsClientId=self._client_id)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col1, basePosition=[-(gap * 0.5 + seg1_len * 0.5), half * 0.25, wall_height], physicsClientId=self._client_id)
        # right segment
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col1, basePosition=[(gap * 0.5 + seg1_len * 0.5), half * 0.25, wall_height], physicsClientId=self._client_id)

    def _spawn_agent(self) -> int:
        assert self._client_id is not None
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.cfg.agent_radius, physicsClientId=self._client_id)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.cfg.agent_radius, rgbaColor=[0.2, 0.8, 0.3, 1], physicsClientId=self._client_id)
        start = [0.0, -(self.cfg.world_size * 0.5), self.cfg.ray_height]
        agent_id = p.createMultiBody(
            baseMass=self.cfg.agent_mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=start,
            physicsClientId=self._client_id,
        )
        # Some friction so it doesn't slide forever
        p.changeDynamics(agent_id, -1, lateralFriction=0.9, rollingFriction=0.02, restitution=0.1, physicsClientId=self._client_id)
        return agent_id

    def _spawn_random_object(self) -> int:
        assert self._client_id is not None
        shape_type = int(self._np_random.integers(0, 3))
        size = float(self._np_random.uniform(*self.cfg.object_size_range))
        mass = float(self._np_random.uniform(*self.cfg.object_mass_range))

        # Random position away from agent
        for _ in range(50):
            x = float(self._np_random.uniform(-self.cfg.world_size * 0.75, self.cfg.world_size * 0.75))
            y = float(self._np_random.uniform(-self.cfg.world_size * 0.75, self.cfg.world_size * 0.75))
            if abs(y + self.cfg.world_size * 0.5) > 1.0:
                break

        z = 0.2 + size

        if shape_type == 0:
            # Box
            half = [size, size * 0.8, size * 0.6]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self._client_id)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.8, 0.3, 0.3, 1], physicsClientId=self._client_id)
        elif shape_type == 1:
            # Sphere
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=size, physicsClientId=self._client_id)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=[0.3, 0.4, 0.9, 1], physicsClientId=self._client_id)
        else:
            # Cylinder
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=size * 0.7, height=size * 1.8, physicsClientId=self._client_id)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=size * 0.7, length=size * 1.8, rgbaColor=[0.9, 0.8, 0.3, 1], physicsClientId=self._client_id)

        oid = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, z],
            physicsClientId=self._client_id,
        )
        p.changeDynamics(oid, -1, lateralFriction=0.8, rollingFriction=0.01, restitution=0.25, physicsClientId=self._client_id)
        return oid

    def _get_obs(self) -> np.ndarray:
        assert self._client_id is not None and self._agent_id is not None
        pos, orn = p.getBasePositionAndOrientation(self._agent_id, physicsClientId=self._client_id)
        vel_lin, vel_ang = p.getBaseVelocity(self._agent_id, physicsClientId=self._client_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        yaw_sin, yaw_cos = math.sin(yaw), math.cos(yaw)

        rays = self._ray_sense(np.array(pos, dtype=np.float32), yaw)

        obs = np.concatenate(
            [
                np.array(pos, dtype=np.float32),
                np.array(vel_lin, dtype=np.float32),
                np.array([yaw_sin, yaw_cos], dtype=np.float32),
                np.array([vel_ang[2]], dtype=np.float32),
                rays.astype(np.float32),
            ],
            axis=0,
        )
        return obs.astype(np.float32)

    def _ray_sense(self, agent_pos: np.ndarray, yaw: float) -> np.ndarray:
        assert self._client_id is not None
        n = self.cfg.num_rays
        L = self.cfg.ray_length
        origin = np.array([agent_pos[0], agent_pos[1], self.cfg.ray_height], dtype=np.float32)

        # Rays in 360 degrees around the agent, rotated by current yaw
        angles = (np.arange(n, dtype=np.float32) / n) * (2 * np.pi) + yaw
        ray_from = []
        ray_to = []
        for a in angles:
            ray_from.append(origin.tolist())
            ray_to.append([origin[0] + L * math.cos(float(a)), origin[1] + L * math.sin(float(a)), origin[2]])

        results = p.rayTestBatch(ray_from, ray_to, physicsClientId=self._client_id)
        dists = np.empty((n,), dtype=np.float32)
        for i, r in enumerate(results):
            hit_fraction = r[2]  # 0..1, 1 means no hit
            dist = float(hit_fraction) * L
            dists[i] = dist / L  # normalize 0..1
        return dists

    def _get_info_base(self) -> Dict[str, Any]:
        assert self._client_id is not None and self._agent_id is not None

        # Contacts (agent with anything)
        cps = p.getContactPoints(bodyA=self._agent_id, physicsClientId=self._client_id)
        contact_count = len(cps)
        max_normal_force = 0.0
        if contact_count > 0:
            max_normal_force = float(max(cp[9] for cp in cps))  # normal force

        # Object displacement since last step (sum)
        obj_disp = 0.0
        obj_speed = 0.0
        for oid in self._object_ids:
            pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=self._client_id)
            pos = np.array(pos, dtype=np.float32)
            last = self._last_obj_pos.get(oid, pos)
            obj_disp += float(np.linalg.norm(pos - last))
            self._last_obj_pos[oid] = pos
            vlin, _ = p.getBaseVelocity(oid, physicsClientId=self._client_id)
            obj_speed += float(np.linalg.norm(np.array(vlin, dtype=np.float32)))
        if len(self._object_ids) > 0:
            obj_speed /= len(self._object_ids)

        # Agent speed
        vlin, _ = p.getBaseVelocity(self._agent_id, physicsClientId=self._client_id)
        agent_speed = float(np.linalg.norm(np.array(vlin, dtype=np.float32)))

        # Agent pos for coverage
        agent_pos = p.getBasePositionAndOrientation(self._agent_id, physicsClientId=self._client_id)[0]

        return {
            "contact_count": contact_count,
            "max_normal_force": max_normal_force,
            "object_displacement": obj_disp,
            "mean_object_speed": obj_speed,
            "agent_speed": agent_speed,
            "agent_pos": np.array(agent_pos, dtype=np.float32),
            "curriculum_level": self.curriculum_level,
        }


# ---------------------------
# Intrinsic reward: RND
# ---------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDModule(nn.Module):
    """
    Random Network Distillation:
      target: fixed random network
      predictor: trained to match target output
      intrinsic reward: prediction error (MSE) on next observation (Burda et al., 2018)
    """
    def __init__(self, obs_dim: int, feature_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.target = MLP(obs_dim, hidden, feature_dim)
        self.predictor = MLP(obs_dim, hidden, feature_dim)

        # Freeze target
        for p_ in self.target.parameters():
            p_.requires_grad = False

        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.rew_rms = RunningMeanStd(shape=())

    @torch.no_grad()
    def _normalize_obs(self, obs: np.ndarray) -> torch.Tensor:
        # Update running stats with a batch dimension
        self.obs_rms.update(obs[None, :])
        obs_n = (obs - self.obs_rms.mean) / self.obs_rms.std
        obs_n = np.clip(obs_n, -5.0, 5.0)
        return torch.from_numpy(obs_n.astype(np.float32))

    def compute_error(self, obs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          err_per_sample: shape (1,)
          obs_tensor: normalized obs tensor (1, obs_dim)
        """
        x = self._normalize_obs(obs).unsqueeze(0)  # (1, obs_dim)
        with torch.no_grad():
            tgt = self.target(x)
        pred = self.predictor(x)
        err = F.mse_loss(pred, tgt, reduction="none").mean(dim=1)  # (1,)
        return err, x

    def update_predictor(self, obs: np.ndarray, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        """
        One-step update on a single observation.
        Returns (intrinsic_reward, predictor_loss)
        """
        err, x = self.compute_error(obs)
        # Intrinsic reward is error (detach)
        intrinsic = float(err.detach().cpu().item())

        # Normalize rewards (helps PPO)
        self.rew_rms.update(np.array([intrinsic], dtype=np.float64))
        intrinsic_norm = intrinsic / float(self.rew_rms.std)

        # Train predictor to minimize err
        optimizer.zero_grad(set_to_none=True)
        loss = err.mean()
        loss.backward()
        optimizer.step()

        return intrinsic_norm, float(loss.detach().cpu().item())


class IntrinsicRNDWrapper(gym.Wrapper):
    """
    Wraps an env to replace reward with intrinsic RND novelty reward and trains the predictor online.
    """
    def __init__(
        self,
        env: gym.Env,
        obs_dim: int,
        beta: float = 1.0,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        super().__init__(env)
        self.device = torch.device(device)
        self.rnd = RNDModule(obs_dim=obs_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=lr)
        self.beta = beta

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # RND intrinsic reward computed on *current* obs (you can also use next-obs style; both are common)
        self.rnd.train()
        intrinsic, rnd_loss = self.rnd.update_predictor(obs, self.optimizer)

        reward = float(self.beta * intrinsic)
        info["rnd_reward"] = reward
        info["rnd_loss"] = rnd_loss
        info["rnd_obs_mean"] = float(np.mean(self.rnd.obs_rms.mean))
        info["rnd_obs_std_mean"] = float(np.mean(self.rnd.obs_rms.std))
        info["rnd_rew_std"] = float(self.rnd.rew_rms.std)

        return obs, reward, terminated, truncated, info


# ---------------------------
# World model learner (for diagnostics, not reward)
# ---------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self._idx = 0
        self._size = 0

    def add(self, obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray, done: bool):
        i = self._idx
        self.obs[i] = obs
        self.act[i] = act
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idxs]),
            torch.from_numpy(self.act[idxs]),
            torch.from_numpy(self.next_obs[idxs]),
            torch.from_numpy(self.done[idxs]),
        )

    @property
    def size(self) -> int:
        return self._size


class WorldModel(nn.Module):
    """
    Simple one-step dynamics model: predicts next observation given obs and action.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class WorldModelWrapper(gym.Wrapper):
    """
    Trains a world model online and reports diagnostics in info dict:
      - wm_loss (MSE one-step prediction)
    """
    def __init__(
        self,
        env: gym.Env,
        obs_dim: int,
        action_dim: int,
        capacity: int = 200_000,
        batch_size: int = 256,
        lr: float = 1e-3,
        updates_per_step: int = 1,
        device: str = "cpu",
        warmup: int = 2_000,
    ):
        super().__init__(env)
        self.device = torch.device(device)
        self.buffer = ReplayBuffer(capacity=capacity, obs_dim=obs_dim, action_dim=action_dim)
        self.model = WorldModel(obs_dim=obs_dim, action_dim=action_dim).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.batch_size = batch_size
        self.updates_per_step = updates_per_step
        self.warmup = warmup

        self._last_obs: Optional[np.ndarray] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs.copy()
        info["wm_loss"] = np.nan
        return obs, info

    def step(self, action):
        assert self._last_obs is not None
        act = np.asarray(action, dtype=np.float32).copy()

        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        # Store transition
        self.buffer.add(self._last_obs, act, obs, done)
        self._last_obs = obs.copy()

        # Update world model
        wm_loss_val = np.nan
        if self.buffer.size >= max(self.warmup, self.batch_size):
            self.model.train()
            losses = []
            for _ in range(self.updates_per_step):
                b_obs, b_act, b_next, b_done = self.buffer.sample(self.batch_size)
                b_obs = b_obs.to(self.device)
                b_act = b_act.to(self.device)
                b_next = b_next.to(self.device)

                pred = self.model(b_obs, b_act)
                loss = F.mse_loss(pred, b_next)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                losses.append(loss.detach().cpu().item())
            wm_loss_val = float(np.mean(losses))

        info["wm_loss"] = wm_loss_val
        return obs, reward, terminated, truncated, info


# ---------------------------
# Research logging callback
# ---------------------------

class ResearchCallback(BaseCallback):
    """
    Logs "interesting insights" to TensorBoard:
      - intrinsic reward statistics
      - RND predictor loss
      - world model loss
      - contact and displacement stats (proxy for interaction + causality)
      - state coverage (grid occupancy)
      - curriculum changes
      - text events when "novelty spikes" or first-time interactions occur
      - optional short videos

    This callback assumes a single-environment setup (DummyVecEnv-like),
    but it will work with VecEnv as long as infos are lists.
    """
    def __init__(
        self,
        writer: SummaryWriter,
        log_every_steps: int = 200,
        coverage_grid: int = 24,
        novelty_z: float = 3.0,
        video_every_episodes: int = 50,
        video_len_steps: int = 240,
        enable_video: bool = False,
        curriculum: bool = True,
        curriculum_patience_episodes: int = 20,
        curriculum_low_mean_reward: float = 0.03,
        curriculum_max_level: int = 6,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.writer = writer
        self.log_every_steps = log_every_steps

        self.coverage_grid = coverage_grid
        self.visited = set()

        self.novelty_rms = RunningMeanStd(shape=())
        self.novelty_z = novelty_z
        self.first_contact_logged = False

        self.enable_video = enable_video
        self.video_every_episodes = video_every_episodes
        self.video_len_steps = video_len_steps

        # Curriculum control
        self.curriculum = curriculum
        self.curriculum_patience_episodes = curriculum_patience_episodes
        self.curriculum_low_mean_reward = curriculum_low_mean_reward
        self.curriculum_max_level = curriculum_max_level
        self.episode_mean_rewards: List[float] = []

        self._episode_idx = 0
        self._ep_reward = 0.0
        self._ep_steps = 0

    def _on_training_start(self) -> None:
        self.writer.add_text("meta", "Training started", 0)

    def _on_step(self) -> bool:
        # SB3 provides these in locals
        infos = self.locals.get("infos", None)
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)

        if infos is None:
            return True

        # Handle VecEnv-like (list of infos)
        for i, info in enumerate(infos):
            r = float(rewards[i]) if rewards is not None else float(info.get("rnd_reward", 0.0))
            self._ep_reward += r
            self._ep_steps += 1

            # Scalars per step (throttled)
            if self.num_timesteps % self.log_every_steps == 0:
                self.writer.add_scalar("intrinsic/rnd_reward", float(info.get("rnd_reward", 0.0)), self.num_timesteps)
                self.writer.add_scalar("intrinsic/rnd_loss", float(info.get("rnd_loss", np.nan)), self.num_timesteps)
                self.writer.add_scalar("world_model/loss", float(info.get("wm_loss", np.nan)), self.num_timesteps)
                self.writer.add_scalar("interaction/contact_count", float(info.get("contact_count", 0.0)), self.num_timesteps)
                self.writer.add_scalar("interaction/object_displacement", float(info.get("object_displacement", 0.0)), self.num_timesteps)
                self.writer.add_scalar("interaction/agent_speed", float(info.get("agent_speed", 0.0)), self.num_timesteps)
                self.writer.add_scalar("curriculum/level", float(info.get("curriculum_level", 0.0)), self.num_timesteps)

            # Coverage update
            agent_pos = info.get("agent_pos", None)
            if agent_pos is not None:
                self._update_coverage(agent_pos)

            # "Interesting insights" as events
            rnd_r = float(info.get("rnd_reward", 0.0))
            self._maybe_log_novelty_spike(rnd_r)

            if (not self.first_contact_logged) and int(info.get("contact_count", 0)) > 0:
                self.writer.add_text("events/first_contact", f"First contact at step {self.num_timesteps}", self.num_timesteps)
                self.first_contact_logged = True

            if dones is not None and bool(dones[i]):
                self._episode_idx += 1
                ep_mean_r = self._ep_reward / max(1, self._ep_steps)
                self.episode_mean_rewards.append(ep_mean_r)

                self.writer.add_scalar("episode/return_intrinsic", self._ep_reward, self._episode_idx)
                self.writer.add_scalar("episode/len", self._ep_steps, self._episode_idx)
                self.writer.add_scalar("episode/mean_intrinsic_reward", ep_mean_r, self._episode_idx)
                self.writer.add_scalar("coverage/unique_cells", len(self.visited), self._episode_idx)

                # Curriculum adjustment
                if self.curriculum:
                    self._maybe_advance_curriculum()

                # Optional video logging
                if self.enable_video and (self._episode_idx % self.video_every_episodes == 0):
                    self._log_video_rollout()

                # Reset episode accumulators
                self._ep_reward = 0.0
                self._ep_steps = 0

        return True

    def _update_coverage(self, agent_pos: np.ndarray) -> None:
        # Map x,y into grid cells. We infer world bounds from env (if accessible); otherwise assume [-6,6].
        x, y = float(agent_pos[0]), float(agent_pos[1])
        # Heuristic bounds; can be made exact by reading env.cfg.world_size via self.training_env.get_attr
        bound = 6.0
        gx = int(np.clip((x + bound) / (2 * bound) * self.coverage_grid, 0, self.coverage_grid - 1))
        gy = int(np.clip((y + bound) / (2 * bound) * self.coverage_grid, 0, self.coverage_grid - 1))
        self.visited.add((gx, gy))

    def _maybe_log_novelty_spike(self, rnd_reward: float) -> None:
        # Track novelty distribution and log spikes (agent discovers "new physics regime")
        self.novelty_rms.update(np.array([rnd_reward], dtype=np.float64))
        mean = float(self.novelty_rms.mean)
        std = float(self.novelty_rms.std)
        if std < 1e-6:
            return
        z = (rnd_reward - mean) / std
        if z > self.novelty_z and self.num_timesteps > 1_000:
            self.writer.add_text(
                "events/novelty_spike",
                f"Novelty spike z={z:.2f}, reward={rnd_reward:.4f} at step {self.num_timesteps}",
                self.num_timesteps,
            )

    def _maybe_advance_curriculum(self) -> None:
        if len(self.episode_mean_rewards) < self.curriculum_patience_episodes:
            return
        recent = self.episode_mean_rewards[-self.curriculum_patience_episodes:]
        recent_mean = float(np.mean(recent))

        # If the agent is "bored" (low novelty) for a while, increase complexity.
        if recent_mean < self.curriculum_low_mean_reward:
            # Get current level from env info via training_env
            try:
                # Works for VecEnv: get_attr returns list; for plain env might raise
                levels = self.training_env.get_attr("curriculum_level")
                current_level = int(levels[0])
            except Exception:
                current_level = 0

            new_level = min(current_level + 1, self.curriculum_max_level)
            if new_level > current_level:
                try:
                    self.training_env.env_method("set_curriculum_level", new_level)
                    self.writer.add_text(
                        "events/curriculum",
                        f"Curriculum advanced {current_level} -> {new_level} at episode {self._episode_idx} (recent_mean={recent_mean:.4f})",
                        self._episode_idx,
                    )
                except Exception:
                    # Fallback: ignore if env doesn't support env_method
                    pass

    def _log_video_rollout(self) -> None:
        """
        Runs a short deterministic rollout (no learning) and logs video.
        """
        try:
            env = self.training_env
            obs = env.reset()
            frames = []
            for _ in range(self.video_len_steps):
                action, _ = self.model.predict(obs, deterministic=False)
                obs, _, dones, infos = env.step(action)
                # render returns array per env if VecEnv; handle both
                frame = None
                try:
                    frame = env.render()
                    if isinstance(frame, list):
                        frame = frame[0]
                except Exception:
                    frame = None
                if frame is not None:
                    frames.append(frame)
                if bool(dones[0]):
                    break
            if len(frames) > 0:
                # TensorBoard video: (N,T,C,H,W) uint8
                video = np.stack(frames, axis=0)  # (T,H,W,C)
                video = np.transpose(video, (0, 3, 1, 2))  # (T,C,H,W)
                video = video[None, ...]  # (1,T,C,H,W)
                self.writer.add_video("videos/rollout", video, global_step=self._episode_idx, fps=30)
        except Exception as e:
            self.writer.add_text("events/video_error", f"Video logging failed: {e}", self._episode_idx)

    def _on_training_end(self) -> None:
        self.writer.add_text("meta", "Training ended", self.num_timesteps)
        self.writer.flush()


# ---------------------------
# Train entrypoint
# ---------------------------

def make_env(cfg: WorldConfig, seed: int, gui: bool, render_mode: Optional[str]) -> gym.Env:
    env = BulletWorldEnv(cfg=cfg, gui=gui, render_mode=render_mode, seed=seed)
    # Train world model (diagnostics) and compute intrinsic RND reward
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env = WorldModelWrapper(env, obs_dim=obs_dim, action_dim=action_dim, device="cpu")
    env = IntrinsicRNDWrapper(env, obs_dim=obs_dim, beta=1.0, lr=1e-4, device="cpu")
    env = Monitor(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/tabula_rasa_bullet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--render_mode", type=str, default=None, choices=[None, "rgb_array", "human"])
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    set_random_seed(args.seed)

    writer = SummaryWriter(log_dir=args.logdir)
    writer.add_text("config", str(vars(args)), 0)

    cfg = WorldConfig()

    env = make_env(cfg, seed=args.seed, gui=args.gui, render_mode="rgb_array" if args.video else None)

    # PPO policy: random init; learns purely from intrinsic reward from wrapper.
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.logdir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=args.seed,
    )

    callback = ResearchCallback(
        writer=writer,
        log_every_steps=200,
        enable_video=args.video,
        video_every_episodes=40,
        video_len_steps=240,
        curriculum=True,
        curriculum_patience_episodes=20,
        curriculum_low_mean_reward=0.03,
        curriculum_max_level=6,
        verbose=0,
    )

    model.learn(total_timesteps=args.total_steps, callback=callback)

    # Save model
    model_path = os.path.join(args.logdir, "ppo_policy.zip")
    model.save(model_path)
    writer.add_text("meta", f"Saved PPO policy to {model_path}", callback.num_timesteps)

    # Also save intrinsic/world-model modules if desired:
    # env.env.env  ... wrappers nesting is env(Monitor)->IntrinsicRNDWrapper->WorldModelWrapper->BulletWorldEnv
    try:
        rnd_wrapper = env.env  # Monitor.env
        torch.save(rnd_wrapper.rnd.state_dict(), os.path.join(args.logdir, "rnd.pt"))
        wm_wrapper = rnd_wrapper.env
        torch.save(wm_wrapper.model.state_dict(), os.path.join(args.logdir, "world_model.pt"))
    except Exception:
        pass

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
