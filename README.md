<div align="center">

# iMFPO

### Induced Mean Flow Policy Optimization for Fast Quadruped Locomotion

Isaac Lab implementation of a one-step MeanFlow policy for Unitree Go2, G1, H1, and Spot velocity control.

</div>

---

## ✨ Overview

iMFPO learns a continuous-time policy flow and evaluates it with **one function evaluation (1 NFE)** at deployment. The training objective combines induced MeanFlow regression with the regression-induced policy ratio `ρF`; a frozen PPO teacher can provide action distillation during training.

The released recipe contains:

- one-step MeanFlow inference for velocity-conditioned locomotion;
- PPO teacher distillation during training (`PPO-KD`);
- Isaac Lab tasks for Go2, G1, H1, and Spot;
- training, evaluation, visualization, and ONNX export utilities.

The PPO teacher is used only while training. The deployed iMFPO actor runs once on the full interval `(r, t) = (0, 1)`.

## 🧩 Method

```text
velocity command + proprioception
                │
                ▼
      one-step MeanFlow actor
                │
                ▼
       12-D joint-position action
                │
                ▼
          PD locomotion control
```

For the paper recipe, Go2, G1, H1, and Spot use the velocity-based ratio `ρF`. The PPO teacher supplies the optional training-time distillation target; it is not required by the exported policy.

## 📦 Repository structure

| Path | Description |
| --- | --- |
| `source/isaaclab_fpo` | MeanFlow actor, iMFPO objective, runner, and export support |
| `source/unitree_rl_lab` | Unitree locomotion task definitions |
| `scripts/rsl_rl` | PPO teacher training |
| `scripts/fpo` | iMFPO training, evaluation, playback, and ONNX export |
| `tests` | CPU smoke and MeanFlow regression tests |

## ⚙️ Environment setup

The repository expects an existing Isaac Lab / Isaac Sim installation. Robot assets and Isaac Lab are not vendored.

```bash
git clone https://github.com/AIGeeksGroup/iMFPO.git
cd iMFPO

source /path/to/isaaclab_experiments/source_env.sh
export ISAAC_ROOT=/path/to/isaaclab_experiments
export PYTHONPATH="$PWD/source/isaaclab_fpo:$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
export UNITREE_MODEL_DIR=/path/to/unitree_model  # required for G1 / H1 USD assets

python -m pip install -e source/isaaclab_fpo
python -m pip install -e source/unitree_rl_lab
```

Use the Isaac Lab environment's Python interpreter for all commands below.

## 🔥 Training

### 1. Train a PPO teacher

```bash
GPU=0 TASK=Unitree-Go2-Velocity MAX_ITERS=3000 \
  bash scripts/run_ppo_teacher.sh
```

### 2. Train iMFPO

```bash
GPU=0 \
TEACHER_CHECKPOINT=/path/to/ppo_teacher.pt \
  bash scripts/run_imfpo.sh
```

Other supported robots use the corresponding task and teacher checkpoint:

```bash
GPU=0 TASK=Unitree-G1-29dof-Velocity \
TEACHER_CHECKPOINT=/path/to/g1_ppo_teacher.pt \
  bash scripts/run_imfpo.sh
```

Useful overrides include `NUM_ENVS`, `MAX_ITERS`, `SEED`, `RUN_NAME`, `WORKDIR`, `RESUME=1`, and `CHECKPOINT=/path/to/model.pt`.

## 📊 Evaluation and playback

```bash
python scripts/fpo/eval_imfpo.py --help
python scripts/fpo/eval_rsl_ppo.py \
  --model_checkpoint /path/to/ppo_teacher.pt

python scripts/fpo/play.py --help
```

For a live Viser scene, use the playback script with `--viser`; the scene contains the robot meshes and the white ground grid used for visual inspection.

## 🚀 Export for deployment

Export a trained iMFPO checkpoint to ONNX:

```bash
python scripts/fpo/export_onnx.py \\
  --checkpoint /path/to/imfpo/model_2499.pt \\
  --sampling-steps 1 \\
  --out-dir /path/to/exported
```

The deployment contract for Go2 is:

| Item | Value |
| --- | --- |
| Observation | 45-D proprioceptive state and velocity command |
| Action | 12-D joint-position action |
| Inference | MeanFlow, 1 NFE |
| Control rate | 50 Hz (`step_dt = 0.02`) |
| PD gains | `kp = 25`, `kd = 0.5` |

The exported ONNX file must be used together with the matching deployment configuration, including joint order, default pose, action scale, and observation normalization.

## 🧪 Verification

Run the CPU regression tests:

```bash
python tests/test_meanflow.py
python tests/smoke_runner_mock_env.py
```

Before deploying to hardware, validate the exported ONNX in a closed-loop simulator and check the joint mapping and PD configuration against the target robot.

## 📄 License and citation

See [LICENCE](LICENCE) for the license. If you use this code or the iMFPO method, please cite the accompanying iMFPO paper and link this repository in your work.

## 🙏 Acknowledgements

This project builds on Isaac Lab, Isaac Sim, RSL-RL, and the Unitree locomotion environments. We thank their authors and maintainers for the open-source infrastructure.
