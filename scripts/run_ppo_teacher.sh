#!/usr/bin/env bash
# Train an RSL-RL PPO teacher for iMFPO KD.
# Usage:
#   GPU=0 bash scripts/run_ppo_teacher.sh
#   GPU=0 TASK=Unitree-G1-29dof-Velocity MAX_ITERS=3000 bash scripts/run_ppo_teacher.sh
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ROOT="${ISAAC_ROOT:-}"
if [[ -z "$ISAAC_ROOT" || ! -f "$ISAAC_ROOT/source_env.sh" ]]; then
  echo "Set ISAAC_ROOT to the Isaac Lab experiment root containing source_env.sh." >&2
  exit 2
fi
set +u
source "${ISAAC_ROOT}/source_env.sh"
set -eo pipefail

export PYTHONPATH="${ROOT}/source/unitree_rl_lab:${ROOT}/scripts/rsl_rl:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

GPU="${GPU:-0}"
TASK="${TASK:-Unitree-Go2-Velocity}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-3000}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-ppo_teacher_s${SEED}}"
WORKDIR="${WORKDIR:-${ROOT}/work/ppo_teacher_gpu${GPU}}"
mkdir -p "$WORKDIR" "${ROOT}/logs/rsl_rl"
cd "$WORKDIR"
unset CUDA_VISIBLE_DEVICES || true

python -u "${ROOT}/scripts/rsl_rl/train.py" \
  --task "$TASK" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --seed "$SEED" \
  --run_name "$RUN_NAME" \
  --device "cuda:${GPU}"
