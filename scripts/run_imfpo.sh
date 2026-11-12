#!/usr/bin/env bash
# Train paper iMFPO on a Unitree velocity task.
# Usage:
#   GPU=0 bash scripts/run_imfpo.sh
#   GPU=0 TASK=Unitree-G1-29dof-Velocity TEACHER_CHECKPOINT=/path/model.pt bash scripts/run_imfpo.sh
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

export PYTHONPATH="${ROOT}/source/isaaclab_fpo:${ROOT}/source/unitree_rl_lab:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

GPU="${GPU:-0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-2500}"
TASK="${TASK:-Unitree-Go2-Velocity}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-imfpo_s${SEED}}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${UNITREE_GO2_PPO_TEACHER:-}}"
if [[ -z "$TEACHER_CHECKPOINT" || ! -f "$TEACHER_CHECKPOINT" ]]; then
  echo "Set TEACHER_CHECKPOINT to a trained PPO teacher .pt (see scripts/run_ppo_teacher.sh)." >&2
  exit 2
fi
WORKDIR="${WORKDIR:-${ROOT}/work/imfpo_${TASK##*-}_gpu${GPU}}"
mkdir -p "$WORKDIR" "${ROOT}/logs/fpo"
cd "$WORKDIR"
unset CUDA_VISIBLE_DEVICES || true

EXTRA=()
[[ -n "${EXPERIMENT_NAME:-}" ]] && EXTRA+=(--experiment_name "$EXPERIMENT_NAME")
[[ "${RESUME:-0}" == "1" ]] && EXTRA+=(--resume)
[[ -n "${CHECKPOINT:-}" ]] && EXTRA+=(--checkpoint "$CHECKPOINT")
[[ -n "${TEACHER_KD_COEF:-}" ]] && EXTRA+=(--teacher_kd_coef "$TEACHER_KD_COEF")
[[ -n "${TEACHER_AUX_ZERO_X0_PROB:-}" ]] && EXTRA+=(--teacher_aux_zero_x0_prob "$TEACHER_AUX_ZERO_X0_PROB")
[[ -n "${LEARNING_RATE:-}" ]] && EXTRA+=(--learning_rate "$LEARNING_RATE")
[[ "${DISABLE_POST_TRAINING_EVAL:-0}" == "1" ]] && EXTRA+=(--disable_post_training_eval)

python -u "${ROOT}/scripts/fpo/train.py" \
  --task "$TASK" \
  --headless \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --seed "$SEED" \
  --fpo_variant imfpo \
  --run_name "$RUN_NAME" \
  --device "cuda:${GPU}" \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" \
  "${EXTRA[@]}"
