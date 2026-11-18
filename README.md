<div align="center">

# PD-SWM

**Spiking World Models with Persistent Dendritic Neurons**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](#installation)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white)](#installation)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

[Overview](#overview) •
[Method](#method) •
[Installation](#installation) •
[Training](#training) •
[Models](#models-and-config-blocks) •
[Baselines](#external-baselines) •
[Layout](#repository-layout)

</div>

## Overview

PD-SWM is an SNN world model for model-based reinforcement learning in partially observable environments.
Its sequence model is built from **Persistent Dendritic Neurons (PDNs)**. The dendrites store the memory, and the
soma fires the spikes.

SNN world models approach the control performance of ANN world models, but they struggle to recall cues from the
distant past. Two problems of existing spiking neurons cause this:

- **Evolution of the state over time.** Distant inputs leave traces that are too weak, and the times at which
  inputs occurred are hard to tell apart.
- **Spike readout.** Resetting the state at firing overwrites the stored memory.

A PDN solves the first problem with **Persistent Dendritic Memory (PDM)** and a **Normalized Write Gain (NWG)**,
and the second with a **Stateless Somatic Readout (SSR)**.

<p align="center">
  <img src="assets/overview.png" alt="PD-SWM overview" width="100%">
</p>
<p align="center"><sub>
  <b>(a)</b> A persistent dendritic neuron unrolled over three environment steps: integrator (blue) and resonant
  (orange) dendritic channels carry their state across steps, and the soma reads them out as spikes.
  <b>(b)</b> PD-SWM: a DreamerV3-style world model whose sequence model is a PDN core.
</sub></p>

## Method

A PDN layer has $N$ neurons with $M$ dendritic channels each and is updated once per environment step.
The firing rates of the input spikes $x_t$ and of the layer's own spikes $\bar S_{t-1}$ are written into the
dendrites through the synapses $B$ and $R$:

```math
I_t = B\,x_t + R\,\bar S_{t-1}
```

<table>
<tr><td width="33%" valign="top">

**Persistent Dendritic Memory**

Each dendritic channel is a damped oscillator.

```math
d_t = \lambda e^{i\theta} d_{t-1} + g(\lambda)\, I_t
```

with $\lambda = e^{-1/\tau}$. The retention time $\tau$ (initialized between 1 and 1000 environment steps) sets
how long a channel remembers. The phase $k\theta$ of a trace marks how long ago its input arrived. Firing never
resets $d_t$, and it keeps evolving during imagined rollouts.

</td><td width="33%" valign="top">

**Normalized Write Gain**

With the leaky write $g = 1-\lambda$, channels with long retention store weak traces. NWG gives every channel
unit energy:

```math
g(\lambda) = \sqrt{1-\lambda^2}
```

Distant traces then stay strong enough to drive the readout. $g$ is computed from the learned $\lambda$ during
training and is a constant at inference.

</td><td width="33%" valign="top">

**Stateless Somatic Readout**

The soma integrates the dendrites with the current input and fires over the $T$ simulation steps with $T$
increasing thresholds:

```math
u_t = \textstyle\sum_{m} \mathrm{Re}(c_m \odot d_{t,m}) + D\,x_t
```

```math
S_t[n] = \Theta\!\left(u_t - n V_{\mathrm{th}}/T\right)
```

The soma has no state to reset, so firing leaves the memory intact.

</td></tr>
</table>

The rest of the agent follows DreamerV3: a spiking encoder and decoder, a latent posterior and prior, reward and
continuation heads, and an actor and critic trained in imagination. All modules are SNNs. The actor and critic
also read the dendritic state as ON/OFF spikes.

| Component | Code | Config block |
|---|---|---|
| Two-layer PDN core with persistent dendrites, PDM and NWG | `networks.PDRNN` | `ours` |
| Half of each neuron's channels resonant, half integrators | `PDRNN(resonant=0.5)` | `resonanthalf` |
| Two dendritic channels per neuron | `PDRNN(channels=2)` | `pdch2` |
| Stateless soma, spike count over $T=5$ steps | `PDRNN(soma='thermo')` | `somathermo` |
| Spike feedback $R\,\bar S_{t-1}$ into the dendrites | `PDRNN(spike_feedback=True)` | `pdfeedback` |
| Actor and critic read the dendrites as ON/OFF spikes | `policy_mem='onoff'` | `policymem` |
| Rank-64 weights from these spikes to the actor and critic | `ac_rank=64` | `aclowrank` |

## Installation

```bash
git clone https://github.com/wxqnl/PD-SWM.git && cd PD-SWM
conda env create -f environment.yml      # Python 3.10, PyTorch 2.4.1, CUDA 12.4
conda activate pd-swm
pip install -r requirements.txt
```

DeepMind Control from pixels renders with OSMesa (`MUJOCO_GL=osmesa`, set by `scripts/train.sh`).

<details>
<summary><b>Memory Maze data</b></summary>

The official Memory Maze 9x9 offline dataset (Pasukonis et al., 2023), read directly from its zip archives:

```bash
bash scripts/data/download_memory_maze.sh eval train-part0 train-part1 ... train-part9
# -> data/memory_maze_9x9_official
```

</details>

<details>
<summary><b>Minecraft Diamond</b></summary>

MineRL 0.4.4 (Minecraft 1.11.2) needs Python 3.11, Java 8 and an X display (Xvfb), so it has its own environment
with the patched MineRL build of dreamerv3-torch:

```bash
conda create -n pd-swm-mc -c conda-forge python=3.11 openjdk=8
conda activate pd-swm-mc
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install https://github.com/NM512/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl \
  gym==0.23.1 numpy==1.23.5 opencv-python==4.10.0.84 ruamel.yaml==0.17.4 einops==0.8.0 cloudpickle==3.0.0 tensorboard
```

On machines with many cores, `scripts/pin_minerl_cpus.py` keeps the MineRL JVMs off the cores of the training
processes, and `scripts/clean_minerl_tmp.py` removes the copies of Minecraft that crashed clients leave in `/tmp`.

</details>

## Training

`scripts/train.sh <task> [seed] [logdir]` runs PD-SWM with the settings of our experiments:

```bash
bash scripts/train.sh bsuite_memlen-101             # BSuite memory length 101 (protocol of R2I), 1M env steps
bash scripts/train.sh popgym_RepeatPreviousMedium   # POPGym RepeatPrevious, 2M env steps
bash scripts/train.sh dmc_walker_walk               # DMC walker from proprioception, 500k env steps
bash scripts/train.sh dmcpix_walker_walk            # DMC walker from pixels, 1M env steps
bash scripts/train.sh bsuite_memlen-64-recall       # world model alone: recall of a cue 64 steps back
bash scripts/train.sh memory_maze_9x9               # Memory Maze offline probing, 1M updates
PYTHON=$(which python) bash scripts/run_minecraft.sh 0   # Minecraft Diamond on GPU 0, 100M env steps (pd-swm-mc)
```

| Task | Benchmark | Budget |
|---|---|---|
| `bsuite_memlen-{8,16,32,64}` | BSuite memory length | 150k to 500k env steps |
| `bsuite_memlen-{51,71,101}` | BSuite memory length, protocol of R2I | 1M env steps |
| `popgym_RepeatPrevious{Easy,Medium}` | POPGym RepeatPrevious | 2M env steps |
| `dmc_{walker_walk,cheetah_run,cartpole_swingup}` | DeepMind Control from proprioception | 500k env steps |
| `dmcpix_walker_walk` | DeepMind Control from pixels | 1M env steps |
| `bsuite_memlen-{8,16,32,64,101}-recall` | world-model probe: recall of a cue $L$ steps back | 12k updates |
| `memory_maze_9x9` | Memory Maze 9x9 offline probing | 1M updates |
| `minecraft_diamond` | Minecraft Diamond (via `scripts/run_minecraft.sh`) | 100M env steps |

Each task maps to one `dreamer.py` command, for example

```bash
python dreamer.py --configs bsuite ours resonanthalf policymem aclowrank somathermo pdfeedback pdch2 rewardbyact \
  --task bsuite_memlen-101 --steps 1e6 --train_ratio 64 --reward_head symlog_mse --batch_length 128 \
  --eval_every 20000 --env_seed_per_env True --seed 0 --logdir runs/pd-swm/bsuite_memlen-101/s0
```

A run writes `metrics.jsonl` (training and evaluation metrics), TensorBoard events and `latest_model.pt` to its
logdir and resumes from them. `probe_mm.py` writes `model.pt` and resumes from it.

## Models and config blocks

`--configs` takes the benchmark block (`bsuite`, `popgym`, `dmc_proprio`, `dmc_vision`, `minecraft`, `mmprobe`)
followed by the blocks of the model. PD-SWM uses the same configuration on every task. All models share the world
model and differ only in the recurrent core of the sequence model:

| Model | Blocks | Core |
|---|---|---|
| **PD-SWM** | `ours resonanthalf policymem aclowrank somathermo pdfeedback pdch2` | `PDRNN` |
| Spiking-WM (Sun et al., 2025) | none | `MCRNN` |
| LIF | `lifcell` | `LIFRNN` |
| GRSN (Qin et al., 2025) | `grsn` | `GRSNCell` |
| DH-SNN (Zheng et al., 2024) | `dhsnn` | `DHSNNCell` |
| TC-LIF (Zhang et al., 2024) | `tclif` | `TCLIFCell` |
| DreamerV3 (ANN) | `grucell annwm` | `GRUCtrl` |
| Spiking-WM with reset interventions | `persist nodendreset taud128 dendgain` | `MCRNN` |

To train a baseline, replace the model blocks of a command, e.g.
`python dreamer.py --configs bsuite tclif rewardbyact --task bsuite_memlen-101 ...`.

- On BSuite every model adds `rewardbyact` (reward head indexed by the action).
- On Memory Maze, which has no actor or critic, PD-SWM drops `aclowrank`.
- The world-model probes (`probe_wm.py`) use the world-model blocks `ours resonanthalf somathermo pdfeedback pdch2`.
- Ablations of PD-SWM: `pdreset` (dendrites cleared at every environment step), `pdeuler` (write gain
  $1-\lambda$ instead of NWG), `pdfixtau` (fixed retention times), `pdsomalif` (soma with a membrane that is reset
  by its spike); see the comments in `configs.yaml`.

## External baselines

<details>
<summary><b>Recall to Imagine (R2I)</b></summary>

Clone [Recall2Imagine](https://github.com/chandar-lab/Recall2Imagine) (commit `6e317e7`) into
`external/Recall2Imagine`, install its environment and apply the action-repeat fix (without it every DMC episode
after the first runs at action repeat 1):

```bash
git -C external/Recall2Imagine apply ../../scripts/patches/r2i_action_repeat.patch
R2I_PYTHON=<its python> bash scripts/run_r2i.sh popgym gym_popgym:popgym-RepeatPreviousEasy-v0 2e6 <gpu> <logdir>
```

DMC: `"dmc_proprio small" dmc_walker_walk 2.5e5` or `dmc_vision dmc_walker_walk 5e5` (500k / 1M frames; `small`
sets the widths of the other models). Memory Maze:
`$R2I_PYTHON probe_mm_r2i.py --configs mmaze small --logdir <logdir> --mm_updates 1000000`.
Minecraft: also apply `scripts/patches/r2i_minecraft.patch`, install MineRL 0.4.4 into the R2I environment, and run
`R2I_PYTHON=<its python> JAVA_HOME=<env with Java 8> bash scripts/run_r2i_minecraft.sh <gpu> [seed]`.

</details>

## Repository layout

```
.
├── dreamer.py             RL training entry (DreamerV3 loop)
├── models.py              world model, actor and critic, imagination
├── networks.py            RSSM and its recurrent cores: PDRNN (PD-SWM), MCRNN (Spiking-WM), LIFRNN,
│                          GRSNCell, DHSNNCell, TCLIFCell, GRUCtrl; spiking encoder and decoder
├── node.py, surrogate.py  spiking neurons (LIF, multicompartment) and surrogate gradients
├── normalization.py       normalization layers (PopNorm)
├── tools.py, exploration.py, parallel.py
│                          logging, replay, environment workers (from dreamerv3-torch)
├── probe_wm.py            world-model memory probe: cue recall on BSuite and POPGym
├── probe_mm.py            Memory Maze offline probing benchmark
├── probe_mm_r2i.py        the same benchmark for R2I (official JAX code, R2I's environment)
├── mm_data.py             Memory Maze offline trajectories and TBTT streams (numpy only)
├── energy.py              synaptic operations per environment step of a trained agent
├── configs.yaml           defaults and named config blocks
├── envs/                  BSuite memory length, POPGym, DeepMind Control, Minecraft Diamond
├── scripts/
│   ├── train.sh           PD-SWM on one task with the settings of our experiments
│   ├── run_minecraft.sh   one model on Minecraft Diamond (X display, relaunch after a crash)
│   ├── run_r2i.sh         R2I baseline (official code) on POPGym and DMC
│   ├── run_r2i_minecraft.sh  R2I on Minecraft Diamond
│   ├── eval_popgym*.py    re-evaluates trained POPGym agents, latent sampled or at its mode (ours and R2I)
│   ├── prune_episodes.py  deletes saved episodes older than the replay of long runs (disk)
│   ├── clean_minerl_tmp.py, pin_minerl_cpus.py   MineRL housekeeping on many-core machines
│   ├── patches/           fixes applied to the R2I code
│   └── data/              Memory Maze download and preparation
└── assets/                README figure
```

`data/`, `external/` and `runs/` are created locally and are not tracked.

## Acknowledgements

The code builds on [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) and on the code of
[Spiking-WM](https://github.com/Brain-Cog-Lab/Spiking-WM) (Sun et al., PNAS 2025), which is also our main
baseline. We thank the authors of Recall2Imagine, Memory Maze, BSuite, POPGym and the DeepMind Control Suite, and
of the GRSN, DH-SNN and TC-LIF neurons, for releasing their code.

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
