# iMFPO

Induced Mean Flow Policy Optimization for Unitree locomotion in Isaac Lab.

This repository ships **only the paper iMFPO recipe** (1-NFE MeanFlow + iMF regression +
frozen PPO action distillation). FPO++ / RFPO / ablations / visual tracks are not included.

## Contents

- `source/isaaclab_fpo`: iMFPO algorithm, MeanFlow actor, runner
- `source/unitree_rl_lab`: Go2 / G1 / H1 / Spot velocity tasks
- `scripts/rsl_rl`: train a native PPO teacher
- `scripts/fpo`: train / eval / play iMFPO
- `tests/`: CPU MeanFlow regression tests

## Setup

```bash
source /path/to/isaaclab_experiments/source_env.sh
export ISAAC_ROOT=/path/to/isaaclab_experiments
export PYTHONPATH="$PWD/source/isaaclab_fpo:$PWD/source/unitree_rl_lab:${PYTHONPATH:-}"
export UNITREE_MODEL_DIR=/path/to/unitree_model   # required for G1 / H1 USD

python -m pip install -e source/isaaclab_fpo
python -m pip install -e source/unitree_rl_lab
```

## 1. Train PPO teacher

```bash
GPU=0 TASK=Unitree-Go2-Velocity MAX_ITERS=3000 bash scripts/run_ppo_teacher.sh
```

## 2. Train iMFPO

```bash
GPU=0 TEACHER_CHECKPOINT=/path/to/ppo_teacher.pt bash scripts/run_imfpo.sh

# Other robots
GPU=0 TASK=Unitree-G1-29dof-Velocity \
  TEACHER_CHECKPOINT=/path/to/g1_ppo_teacher.pt \
  bash scripts/run_imfpo.sh
```

## 3. Evaluate

```bash
python scripts/fpo/eval_imfpo.py --help
python scripts/fpo/eval_rsl_ppo.py --model_checkpoint /path/to/ppo_teacher.pt
```

## CPU tests

```bash
python tests/test_meanflow.py
```

## Notes

- Deployed iMFPO policies are **1 NFE**. The PPO teacher is used only during training KD.
- Training writes under `work/` by default (`WORKDIR` overrides).
- Robot assets and Isaac Lab / Isaac Sim are not vendored.
