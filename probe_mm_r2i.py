"""Memory Maze offline probing benchmark for R2I (Samsami et al., ICLR 2024): probe_mm.py with R2I's world model.

Runs in R2I's JAX environment (R2I_PYTHON) on its official code (external/Recall2Imagine, commit 6e317e7):

  model    agent.WorldModel of the official code with R2I's own Memory Maze configuration (`mmaze`: S3M core with
           5 SSM layers, SSM state 512) at the widths of the other models (R2I's `small` block: deter, units and
           MLPs 512, CNN depth 32; 22.0M parameters, ours 16.6M; `mmaze` alone is 69.5M), trained with its own loss
           and optimizer. The batch follows the benchmark protocol shared by every model of the table (configs.yaml
           `mmprobe`): 32 streams of consecutive 48-step chunks, the state carried from one chunk to the next (TBTT),
           1M updates.
  probe    the MLP of probe_mm.py (4 x 1024, LayerNorm eps 1e-3, ELU, PyTorch's default initialisation), in float32,
           reads the stop-gradient features R2I's actor reads on Memory Maze (actor.inputs: deter, stoch, hidden)
           plus the agent position and orientation; trained on the same batches with Adam (lr 3e-4, eps 1e-5).
           The no-memory probe reads the image embedding.
  data     the same trajectories, streams and scoring as probe_mm.py (mm_data.py). The dataset stores the action
           that led to each image; R2I's replay stores the action taken after it, so the actions are shifted by one
           step (R2I's world model then conditions step t on the same previous action as the other models).

usage: $R2I_PYTHON probe_mm_r2i.py [--configs mmaze small] --logdir runs/r2i_agent/memory_maze_9x9/R2I/s0 \
           --mm_updates 1000000 [R2I config overrides, e.g. --seed 0]
"""
import argparse
import concurrent.futures
import json
import os
import pathlib
import pickle
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
R2I = ROOT / "external" / "Recall2Imagine"
sys.path[:0] = [str(ROOT), str(R2I), str(R2I / "recall2imagine")]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # the GPUs are shared
os.environ.setdefault("WANDB_MODE", "disabled")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import embodied  # noqa: E402
from recall2imagine import agent as agt  # noqa: E402
from recall2imagine import jaxutils, nets  # noqa: E402
from recall2imagine import ninjax as nj  # noqa: E402

import mm_data  # noqa: E402

f32 = jnp.float32
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)  # noqa: E731
N_ACT = 6


def torch_uniform(shape, fan_in):
    """PyTorch's default nn.Linear initialisation (weights and biases)."""
    bound = 1.0 / np.sqrt(fan_in)
    return jax.random.uniform(nj.rng(), shape, f32, -bound, bound)


class ProbeMLP(nj.Module):
    """probe_mm.ProbeMLP: 4 hidden layers of 1024 units, LayerNorm (eps 1e-3), ELU; heads for the 81 wall tiles and
    the 3 object positions. float32 throughout."""

    def __init__(self, units=1024, layers=4):
        self.units, self.layers = units, layers

    def _linear(self, name, x, units):
        w = self.get(f"{name}_w", torch_uniform, (x.shape[-1], units), x.shape[-1])
        b = self.get(f"{name}_b", torch_uniform, (units,), x.shape[-1])
        return x @ w + b

    def __call__(self, x):
        for i in range(self.layers):
            x = self._linear(f"h{i}", x, self.units)
            mean = x.mean(-1, keepdims=True)
            x = (x - mean) * jax.lax.rsqrt(((x - mean) ** 2).mean(-1, keepdims=True) + 1e-3)
            x = x * self.get(f"h{i}_scale", jnp.ones, (self.units,), f32) + self.get(f"h{i}_offset", jnp.zeros, (self.units,), f32)
            x = jax.nn.elu(x)
        return self._linear("walls", x, 81), self._linear("objects", x, 6)


class ProbeAgent(nj.Module):
    """R2I's world model plus the two probes."""

    def __init__(self, obs_space, act_space, config, probe_lr):
        self.config = config
        self.wm = agt.WorldModel(obs_space, act_space, config, name="wm")
        self.feat = nets.Input(config.actor.inputs, dims="deter")  # what the actor reads
        self.probes = {"state": ProbeMLP(name="probe_state"), "no_memory": ProbeMLP(name="probe_nomem")}
        self.opts = {k: jaxutils.Optimizer(probe_lr, opt="adam", eps=1e-5, clip=0.0, name=f"probe_{k}_opt")
                     for k in self.probes}

    def initial(self, batch_size):
        return self.wm.initial(batch_size)

    def preprocess(self, obs):  # agent.Agent.preprocess
        obs = obs.copy()
        for key, value in obs.items():
            if len(value.shape) > 3 and value.dtype == jnp.uint8:
                obs[key] = jaxutils.cast_to_compute(value) / 255.0
            else:
                obs[key] = value.astype(f32)
        obs["cont"] = 1.0 - obs["is_terminal"].astype(f32)
        return obs

    def probe_loss(self, name, feat, probe):
        walls_logit, objects = self.probes[name](jnp.concatenate([feat, probe["pose"]], -1))
        B, L = feat.shape[:2]
        walls = jnp.broadcast_to(probe["walls"][:, None], (B, L, 81))
        bce = jnp.maximum(walls_logit, 0) - walls_logit * walls + jnp.log1p(jnp.exp(-jnp.abs(walls_logit)))
        objects_se = ((objects.reshape(B, L, 3, 2) - probe["targets"]) ** 2).sum(-1).mean(-1)
        walls_acc = ((walls_logit > 0).astype(f32) == walls).astype(f32).mean(-1)
        return bce.mean() + objects_se.mean(), (objects_se, walls_acc)

    def train(self, data, probe, state):
        data = self.preprocess(data)
        state, outs, mets = self.wm.train(data, state)
        feats = {"state": sg(self.feat(outs["post"])).astype(f32), "no_memory": sg(outs["embed"]).astype(f32)}
        out = {k: mets[k] for k in ("model_loss_mean", "image_loss_mean", "reward_loss_mean", "dyn_loss_mean",
                                     "rep_loss_mean") if k in mets}
        for name, feat in feats.items():
            _, (objects_se, walls_acc) = self.opts[name]([self.probes[name]], self.probe_loss, name, feat, probe,
                                                         has_aux=True)
            if name == "state":
                out["train_objects_mse"], out["train_walls_acc"] = objects_se.mean(), walls_acc.mean()
        return state, out

    def evaluate(self, data, probe, state):
        """WorldModel.loss's filtering without the update; per-step probe metrics."""
        data = self.preprocess(data)
        embed = self.wm.encoder(data)
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], 1)
        post, _ = self.wm.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)
        state = ({k: v[:, -1] for k, v in post.items()}, data["action"][:, -1])
        out = {}
        for name, feat in {"state": self.feat(post), "no_memory": embed}.items():
            _, out[name] = self.probe_loss(name, feat.astype(f32), probe)
        return state, out


def next_actions(eps, starts, length):
    """The action taken after each step of the chunks eps[b][starts[b]: starts[b] + length] (zero after the last
    step of a trajectory)."""
    out = np.zeros((len(eps), length, N_ACT), np.float32)
    for b, (ep, p) in enumerate(zip(eps, starts)):
        a = ep["action"][p + 1: p + length + 1]
        out[b, :len(a)] = a
    return out


def r2i_data(batch, action):
    n = batch["reward"].shape
    data = {"image": batch["image"], "action": action, "reward": batch["reward"].astype(np.float32),
            "is_first": batch["is_first"], "is_last": np.zeros(n, bool), "is_terminal": np.zeros(n, bool)}
    probe = {"pose": np.concatenate([batch["agent_pos"], batch["agent_dir"]], -1).astype(np.float32),
             "walls": batch["maze_layout"].reshape(n[0], 81).astype(np.float32),
             "targets": batch["targets_vec"].reshape(*n, 3, 2).astype(np.float32)}
    return data, probe


class Runner:
    """The jitted pure functions of ProbeAgent on one device (jaxagent.JAXAgent, reduced to what is needed here)."""

    def __init__(self, config, probe_lr, seed):
        jaxutils.COMPUTE_DTYPE = getattr(jnp, config.jax.precision)
        jax.config.update("jax_platform_name", config.jax.platform)
        self.dev = jax.devices(config.jax.platform)[0]
        obs_space = {"image": embodied.Space(np.uint8, (64, 64, 3)), "reward": embodied.Space(np.float32),
                     "is_first": embodied.Space(bool), "is_last": embodied.Space(bool),
                     "is_terminal": embodied.Space(bool)}
        act_space = {"action": embodied.Space(np.float32, (N_ACT,), 0, 1), "reset": embodied.Space(bool)}
        self.agent = ProbeAgent(obs_space, act_space, config, probe_lr, name="agent")
        self.rng = np.random.default_rng(seed)
        kw = dict(device=self.dev)
        self._init = nj.jit(nj.pure(lambda x: self.agent.initial(len(x))), **kw)
        self._train = nj.jit(nj.pure(self.agent.train), **kw)
        self._eval = nj.jit(nj.pure(self.agent.evaluate), **kw)
        B, L = config.batch_size, config.batch_length
        dummy = {"image": np.zeros((B, L, 64, 64, 3), np.uint8), "reward": np.zeros((B, L)),
                 "agent_pos": np.zeros((B, L, 2)), "agent_dir": np.zeros((B, L, 2)),
                 "targets_vec": np.zeros((B, L, 3, 2)), "maze_layout": np.zeros((B, 9, 9)),
                 "is_first": np.ones((B, L), bool)}
        data, probe = self.put(r2i_data(dummy, np.zeros((B, L, N_ACT), np.float32)))
        self.varibs = {}
        state, self.varibs = self._init(self.varibs, self.key(), data["is_first"])
        self.varibs = self._train(self.varibs, self.key(), data, probe, state, init_only=True)

    def key(self):
        return jax.device_put(self.rng.integers(2 ** 63 - 1), self.dev)

    def put(self, tree):
        return jax.device_put(tree, self.dev)

    def initial(self, batch_size):
        state, _ = self._init(self.varibs, self.key(), self.put(np.ones(batch_size, bool)))
        return state

    def train(self, data, probe, state):
        (state, mets), self.varibs = self._train(self.varibs, self.key(), *self.put((data, probe)), state)
        return state, mets

    def evaluate(self, data, probe, state):
        (state, out), _ = self._eval(self.varibs, self.key(), *self.put((data, probe)), state)
        return state, out

    def save(self):
        return jax.tree_util.tree_map(np.asarray, jax.device_get(self.varibs))

    def load(self, varibs):
        self.varibs = self.put(varibs)


def evaluate(runner, episodes, idx, length, pool, batch=32):
    """probe_mm.evaluate: filter the trajectories idx from their start; score the second half (last 500 steps).
    Batches are padded to `batch` trajectories (one compiled shape) and the padding is dropped."""
    out = {name: {"objects": [], "walls": []} for name in ("state", "no_memory")}
    for c in range(0, len(idx), batch):
        ids = list(idx[c: c + batch])
        n = len(ids)
        eps = list(pool.map(episodes.load, ids + [ids[-1]] * (batch - n)))
        T = len(eps[0]["reward"])
        state = runner.initial(batch)
        per = {name: {"objects": [], "walls": []} for name in out}
        for p in range(0, T, length):
            chunk = {k: np.stack([ep[k][p: p + length] for ep in eps]) for k in (*mm_data.KEYS, "is_first")
                     if k != "maze_layout"}
            chunk["maze_layout"] = np.stack([ep["maze_layout"] for ep in eps])
            L = chunk["reward"].shape[1]
            state, res = runner.evaluate(*r2i_data(chunk, next_actions(eps, [p] * batch, L)), state)
            for name, (objects_se, walls_acc) in jax.device_get(res).items():
                per[name]["objects"].append(np.asarray(objects_se)[:n])
                per[name]["walls"].append(np.asarray(walls_acc)[:n])
        for name in out:
            for k in ("objects", "walls"):
                out[name][k].append(np.concatenate(per[name][k], 1))
    log = {}
    for name in out:
        for k in ("objects", "walls"):
            m = np.concatenate(out[name][k], 0)[:, 1:]  # the 1000 steps after the initial image
            half = m.shape[1] // 2
            key = f"eval_{k}_mse" if k == "objects" else f"eval_{k}_acc"
            prefix = "" if name == "state" else "nomem_"
            log[prefix + key] = float(m[:, half:].mean())
            log[prefix + key + "_curve"] = [round(float(m[:, b].mean()), 4) for b in np.array_split(np.arange(m.shape[1]), 10)]
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/memory_maze_9x9_official"))
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--mm_updates", type=int, default=1000000)
    parser.add_argument("--mm_eval_every", type=int, default=10000)
    parser.add_argument("--mm_eval_eps", type=int, default=200)
    parser.add_argument("--probe_lr", type=float, default=3e-4)
    parser.add_argument("--log_every", type=int, default=10, help="updates between metric reads (each one syncs)")
    parser.add_argument("--configs", nargs="+", default=["mmaze", "small"],
                        help="R2I config blocks over its defaults; `small` sets the size of the other models")
    args, rest = parser.parse_known_args()
    config = embodied.Config(agt.Agent.configs["defaults"])
    for name in args.configs:
        config = config.update(agt.Agent.configs[name])
    config = config.update({"batch_size": 32, "batch_length": 48, "jax.prealloc": False})  # the benchmark's batch
    config = embodied.Flags(config).parse(rest)
    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    done = logdir / "results.json"
    if done.exists() and json.loads(done.read_text())["step"] >= args.mm_updates:
        print("already finished")
        return
    config.save(str(logdir / "config.yaml"))
    print(config, flush=True)
    train_src, eval_src = mm_data.split_sources(args.data)
    train, evals = mm_data.Episodes(train_src), mm_data.Episodes(eval_src)
    print(f"{len(train)} training and {len(evals)} evaluation trajectories", flush=True)
    runner = Runner(config, args.probe_lr, config.seed)
    params = {k: int(v) for k, v in jaxutils.Optimizer.PARAM_COUNTS.items() if v}
    print("parameters:", params, flush=True)
    pool = concurrent.futures.ThreadPoolExecutor(16)
    eval_all = np.arange(len(evals))
    eval_sub = eval_all[:: max(1, len(evals) // args.mm_eval_eps)][: args.mm_eval_eps]
    start = 0
    if (logdir / "model.pkl").exists():  # resume after an interruption or to extend the budget
        with open(logdir / "model.pkl", "rb") as f:
            ckpt = pickle.load(f)
        runner.load(ckpt["varibs"])
        start = ckpt["update"]
        print(f"resumed at update {start} (the TBTT state restarts)", flush=True)
    metrics = (logdir / "metrics.jsonl").open("a")
    if start == 0:
        log = {"step": 0, "train_trajectories": len(train), "eval_trajectories": len(evals), "params": params,
               **mm_data.constant_baseline(train, evals, eval_all, pool)}
        print(json.dumps(log), flush=True)
        metrics.write(json.dumps(log) + "\n")
    B, L = config.batch_size, config.batch_length
    streams = mm_data.Streams(train, B, L, config.seed + start)
    state = runner.initial(B)
    running, n_since, t0 = {}, 0, time.time()
    for update in range(start + 1, args.mm_updates + 1):
        batch = streams.next()  # advances streams.pos past the chunk; streams.cur holds its trajectories
        action = next_actions(streams.cur, [p - L for p in streams.pos], L)
        state, mets = runner.train(*r2i_data(batch, action), state)
        n_since += 1
        if update % args.log_every == 0:
            for k, v in jax.device_get(mets).items():
                running.setdefault(k, []).append(float(v))
        if update % args.mm_eval_every == 0 or update == args.mm_updates:
            log = {"step": update, **{k: float(np.mean(v)) for k, v in running.items()},
                   "sec_per_update": (time.time() - t0) / n_since}
            log["model_loss"] = log.get("model_loss_mean")
            final = update == args.mm_updates
            log.update(evaluate(runner, evals, eval_all if final else eval_sub, L, pool))
            with open(logdir / "model.pkl.tmp", "wb") as f:
                pickle.dump({"varibs": runner.save(), "update": update}, f)
            os.replace(logdir / "model.pkl.tmp", logdir / "model.pkl")
            print(json.dumps(log), flush=True)
            metrics.write(json.dumps(log) + "\n")
            metrics.flush()
            running, n_since, t0 = {}, 0, time.time()
    (logdir / "results.json").write_text(json.dumps(log, indent=1))
    (logdir / f"results_{args.mm_updates // 1000}k.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
