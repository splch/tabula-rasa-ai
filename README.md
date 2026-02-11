# Tabula Rasa 3D Discovery

Minimal single-file research baseline for intrinsic-motivation RL in a 3D physics world.

<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/d8838e1c-0b2c-4df2-aaf9-59134fb77110" />

## What It Does

- Builds a custom **PyBullet** world with walls, dynamic objects, and curriculum structures.
- Uses a **Gymnasium** environment with continuous actions: `forward_throttle`, `yaw_throttle`.
- Trains an **SB3 PPO** policy from random initialization.
- Replaces task reward with **RND intrinsic reward** (novelty).
- Trains a separate **world model** online for diagnostics.
- Logs learning signals and events to **TensorBoard**.

## Repository Layout

- `tabula_rasa_3d.py`: environment, wrappers, callback, and training entrypoint.
- `requirements.txt`: Python dependencies.
- `runs/`: output directory for training runs.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run training (headless):

```bash
python tabula_rasa_3d.py --total_steps 500000 --logdir runs/exp1
```

Run with PyBullet GUI:

```bash
python tabula_rasa_3d.py --gui --total_steps 500000 --logdir runs/exp_gui
```

Run with TensorBoard video logging enabled:

```bash
python tabula_rasa_3d.py --video --total_steps 500000 --logdir runs/exp_video
```

Open TensorBoard:

```bash
tensorboard --logdir runs
```

## Key CLI Flags

- `--logdir`: output directory (default: `runs/tabula_rasa_bullet`)
- `--seed`: random seed (default: `0`)
- `--total_steps`: PPO timesteps (default: `500000`)
- `--gui`: use PyBullet GUI mode
- `--video`: log rollout videos to TensorBoard

## What Gets Logged

Main scalar groups:

- `intrinsic/*`: RND reward and loss
- `world_model/*`: one-step prediction loss
- `interaction/*`: contact count, object displacement, agent speed
- `coverage/*`: unique visited grid cells
- `episode/*`: return, length, mean intrinsic reward
- `curriculum/*`: current curriculum level

Event text logs:

- `events/first_contact`
- `events/novelty_spike`
- `events/curriculum`

Optional video:

- `videos/rollout` (when `--video` is enabled)

## Saved Artifacts

In each run directory (`--logdir`), training saves:

- `ppo_policy.zip`
- `rnd.pt`
- `world_model.pt`
- TensorBoard event files

## Notes

- There is **no extrinsic task reward** in this setup.
- The world model is for diagnostics only; it is not used as reward.
- Curriculum increases world complexity when recent intrinsic reward stays low.
