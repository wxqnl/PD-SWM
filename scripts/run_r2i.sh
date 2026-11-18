#!/bin/bash
# R2I (Samsami et al., ICLR 2024), official code (external/Recall2Imagine, commit 6e317e7) with one of its own
# configurations (popgym, dmc_proprio, dmc_vision; `small` after the task block sets the widths of the other models:
# deter, units and MLPs 512, CNN depth 32), unchanged except: run.steps (our budget, in R2I's policy steps:
# DMC repeats each action twice, so 2.5e5 / 5e5 policy steps are our 500k / 1M frames), replay_size 2e6 (the
# default FIFO buffer of 1e7 never fills within these budgets, so sampling is the same) and jax.prealloc False
# (the GPUs are shared). One bug fix in the code, scripts/patches/r2i_action_repeat.patch: its ActionRepeat never
# cleared its done flag on reset, so every DMC episode after the first ran at action repeat 1. The `train` script logs the score of the training episodes, as in the R2I paper; its
# evaluation policy is the same (it also samples the action), so there is no separate evaluation.
# usage: R2I_PYTHON=<python of the R2I environment> scripts/run_r2i.sh "<configs>" <task> <steps> <gpu> <logdir>
#   e.g. popgym gym_popgym:popgym-RepeatPreviousEasy-v0 2e6, "dmc_proprio small" dmc_walker_walk 2.5e5,
#        dmc_vision dmc_walker_walk 5e5 (dmc_vision already has these widths)
set -u
config=$1; task=$2; steps=$3; gpu=$4; logdir=$(realpath -m "$5")  # absolute: the script changes directory
mkdir -p "$logdir"
cd "$(dirname "$0")/../external/Recall2Imagine"
echo $$ > "$logdir/.pid"; echo "gpu$gpu" > "$logdir/.gpu"
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  "${R2I_PYTHON:-python}" recall2imagine/train.py \
  --configs $config --task "$task" --logdir "$logdir" \
  --run.steps "$steps" --replay_size 2e6 --jax.prealloc False --seed 0 >> "$logdir/stdout.log" 2>&1  # appends: a rerun resumes from checkpoint.ckpt
echo $? > "$logdir/.exit"
