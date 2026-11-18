#!/bin/bash
# One model on Minecraft Diamond: starts an X display for MineRL (Xvfb), runs dreamer.py with the minecraft
# config and the model's configs, and relaunches it after a crash (a MineRL Java client can die); dreamer.py
# resumes from latest_model.pt and the saved episodes.
#
# usage: bash scripts/run_minecraft.sh <gpu> [seed] [model]
#   model   PD2Lv2r64thermofbch2_tau1000 (PD-SWM, default; the command of scripts/train.sh minecraft_diamond),
#           base_tau2 (Spiking-WM), LIF_tau2, GRSN_tau2, DHSNN, TCLIF, ann_tau2 (DreamerV3); the run names of
#           the other tasks. Logs in runs/minecraft/minecraft_diamond/<model>/s<seed>.
#   PYTHON  python of the env with minerl 0.4.4 (Java 8 in the same env's bin/)
#   XVFB    Xvfb binary (default: Xvfb on PATH)
set -u
cd "$(dirname "$0")/.."
gpu=$1; seed=${2:-0}; model=${3:-PD2Lv2r64thermofbch2_tau1000}
case $model in
  PD2Lv2r64thermofbch2_tau1000) configs="ours resonanthalf policymem aclowrank somathermo pdfeedback pdch2";;
  base_tau2) configs="";;
  LIF_tau2)  configs="lifcell";;
  GRSN_tau2) configs="grsn";;
  DHSNN)     configs="dhsnn";;
  TCLIF)     configs="tclif";;
  ann_tau2)  configs="grucell annwm";;
  *) echo "unknown model: $model" >&2; exit 1;;
esac
logdir=runs/minecraft/minecraft_diamond/$model/s$seed
PY=${PYTHON:-python}
export PATH="$(dirname "$PY"):$PATH" JAVA_HOME="$(dirname "$(dirname "$PY")")"
export DISPLAY=:$((90 + gpu)) CUDA_VISIBLE_DEVICES=$gpu
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
  $PY dreamer.py --configs minecraft $configs --steps 1e8 --task minecraft_diamond --seed "$seed" --logdir "$logdir" \
    && break
  status=$?  # before $(date ...), which resets $?
  echo "$(date '+%F %T') dreamer.py exited with $status; relaunch $try in 60 s" >&2
  sleep 60
done
