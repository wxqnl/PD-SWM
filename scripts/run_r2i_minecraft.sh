#!/bin/bash
# R2I (official code, external/Recall2Imagine) on Minecraft Diamond, in the setting of the other models there.
# R2I has no Minecraft configuration of its own; its code base (DreamerV3) has, and this uses that block: 16 envs,
# train ratio 16, the encoder/decoder keys, inventory logged as the episode maximum. Widths `small` as for DMC
# (deter, units and MLPs 512, CNN depth 32); replay 1e6, the dataset_size of the PyTorch runs; the rest is R2I's
# defaults. Two fixes in its Minecraft env, scripts/patches/r2i_minecraft.patch: options.txt is found from the
# minerl package of this interpreter, and the reward is always float32 (on the first step of an episode every
# reward term is the int 0, and R2I's on-disk replay needs one dtype per key). Relaunches after a crash whatever the
# exit status (train.py catches a lost env worker, a RuntimeError, prints 'Gracefully stopping...' and exits 0);
# train.py resumes from checkpoint.ckpt.
#
# usage: bash scripts/run_r2i_minecraft.sh <gpu> [seed]
#   R2I_PYTHON  python of the R2I env with minerl 0.4.4 and gym 0.23.1 (r2i-mc)
#   JAVA_HOME   env with Java 8 (pd-swm-mc)
#   XVFB        Xvfb binary (default: Xvfb on PATH)
set -u
gpu=$1; seed=${2:-0}
cd "$(dirname "$0")/.."
logdir=$(realpath -m runs/minecraft/minecraft_diamond/R2I/s$seed)
mkdir -p "$logdir"
cd external/Recall2Imagine
export PATH="$(dirname "$R2I_PYTHON"):$JAVA_HOME/bin:$PATH" DISPLAY=:$((90 + gpu)) CUDA_VISIBLE_DEVICES=$gpu
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
# Each MineRL client is a JVM that sizes its GC and compiler threads, and Mesa its llvmpipe render threads, to
# the 128 cores (83 + 32 per client); with 8 runs of 16 clients they oversubscribe the CPU and the client sockets
# time out. Two of each change nothing in the game or the frames.
export JAVA_TOOL_OPTIONS="-XX:ParallelGCThreads=2 -XX:CICompilerCount=2" LP_NUM_THREADS=2
# The same for the CPU thread pools of the Python processes (OpenMP/MKL, 128 threads each by default): 8 runs
# with 16 env workers each made ~44M context switches per second and left the GPUs waiting.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
xvfb=
trap '[ -n "$xvfb" ] && kill $xvfb' EXIT
for try in $(seq 30); do
  if [ -z "$xvfb" ] || ! kill -0 "$xvfb" 2>/dev/null; then  # (re)start the X display; it can die (e.g. /tmp full: no lock file)
    ${XVFB:-Xvfb} $DISPLAY -screen 0 1024x768x24 -nolisten tcp &
    xvfb=$!
    sleep 2
  fi
  "$R2I_PYTHON" recall2imagine/train.py --configs small --task minecraft_diamond --logdir "$logdir" --seed "$seed" \
    --run.steps 1e8 --replay_size 1e6 --jax.prealloc False --envs.amount 16 --run.train_ratio 16 \
    --run.log_keys_max '^log_inventory.*' \
    --encoder.mlp_keys 'inventory|inventory_max|equipped|health|hunger|breath|reward' --encoder.cnn_keys image \
    --decoder.mlp_keys 'inventory|inventory_max|equipped|health|hunger|breath' --decoder.cnn_keys image
  status=$?  # before $(date ...), which resets $?
  echo "$(date '+%F %T') train.py exited with $status; relaunch $try in 60 s" >&2
  sleep 60
done
