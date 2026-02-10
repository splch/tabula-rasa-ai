Below is a **single-file, runnable research baseline** that implements:

* **3D rigid-body physics world** in **PyBullet** (fast to prototype; widely used for RL/robotics simulation) ([GitHub][1])
* **Gymnasium-style environment** (`reset/step`) ([gymnasium.farama.org][2])
* **Point agent** (a sphere) with continuous actions (forward force + yaw torque)
* **No pretraining**: all networks start randomly initialized
* **Intrinsic reward** via **Random Network Distillation (RND)** (novelty = predictor error vs fixed random target network) ([arXiv][3])
* **World-model learning** in parallel (one-step predictor for diagnostics)
* **Research logging** via **TensorBoard SummaryWriter** (scalars + text events + optional videos) ([PyTorch Documentation][4])
* A simple **autocurriculum**: when novelty stays low for long enough, the environment increases complexity (more objects; ramp; second room)

If you want an alternative curiosity mechanism later, you can swap RND for an Intrinsic Curiosity Module-style forward prediction error scheme ([arXiv][5]), but RND is extremely simple and robust to start with ([arXiv][3]).

## What this logs as “interesting insights”

Into TensorBoard, it logs:

* `intrinsic/rnd_reward`, `intrinsic/rnd_loss` (novelty and predictor learning)
* `world_model/loss` (one-step dynamics prediction error; “physics understanding” proxy)
* `interaction/contact_count`, `interaction/object_displacement` (how much the agent is *causing things to happen*)
* `coverage/unique_cells` (state-space exploration coverage)
* `events/first_contact` (first meaningful interaction)
* `events/novelty_spike` (rare high-surprise transitions → often correspond to discovering new interactions/regimes)
* `events/curriculum` (when environment complexity is bumped)

Optionally: `videos/rollout` (short deterministic rollouts) when `--video` is enabled.

---

## How to run

Install deps (minimal set):

```bash
pip install pybullet gymnasium stable-baselines3 torch tensorboard
```

Train (headless):

```bash
python tabula_rasa_3d.py --total_steps 500000 --logdir runs/exp1
```

Train with GUI:

```bash
python tabula_rasa_3d.py --gui --total_steps 500000 --logdir runs/exp_gui
```

Enable video logging (uses `rgb_array` render):

```bash
python tabula_rasa_3d.py --video --total_steps 500000 --logdir runs/exp_video
```

View logs:

```bash
tensorboard --logdir runs
```

---

## Why these components match your “tabula rasa discovery” goal

* **No human task reward**: PPO learns purely from intrinsic novelty (RND) ([arXiv][3]) using SB3’s PPO implementation ([stable-baselines3.readthedocs.io][6])
* **No pretrained perception or semantics**: all networks are random at initialization
* **“Discover rules” is operationalized** as:

  * decreasing **world-model loss** over time (predictive physics competence)
  * decreasing novelty in familiar regimes + novelty spikes when encountering new interactions
  * increasing exploration **coverage**
  * increasing interaction metrics (contacts, object displacement)

---

If you want, I can extend this exact codebase in the most research-useful direction next (without changing the core philosophy): **pixel observations + latent world model**, **object-centric dynamics**, or **vectorized envs + shared intrinsic module** for scale.

[1]: https://raw.githubusercontent.com/bulletphysics/bullet3/master/docs/pybullet_quickstartguide.pdf "https://raw.githubusercontent.com/bulletphysics/bullet3/master/docs/pybullet_quickstartguide.pdf"
[2]: https://gymnasium.farama.org/api/env/ "https://gymnasium.farama.org/api/env/"
[3]: https://arxiv.org/abs/1810.12894 "https://arxiv.org/abs/1810.12894"
[4]: https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html "https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html"
[5]: https://arxiv.org/abs/1705.05363 "https://arxiv.org/abs/1705.05363"
[6]: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html "https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html"
