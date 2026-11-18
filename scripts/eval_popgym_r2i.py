"""Evaluate the final R2I agent of a POPGym run (official code) with the latent sampled or at its mode.

The counterpart of scripts/eval_popgym.py for R2I; runs in the R2I environment (R2I_PYTHON). Loads
checkpoint.ckpt and the run's config.yaml and plays episodes with agent.policy(mode='eval'). R2I's
eval policy samples both the posterior's stochastic state and the action. One variant per process
(the policy is traced once):

  standard  R2I's own evaluation: latent and action sampled
  sampled   greedy action, latent sampled: the standard evaluation of dreamer.py
  mode      greedy action, latent at its mode (deterministic agent)

usage: R2I_PYTHON scripts/eval_popgym_r2i.py --variant mode runs/r2i_agent/popgym_RepeatPreviousEasy/R2I/s0
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
R2I = ROOT / "external" / "Recall2Imagine"
sys.path[0:1] = [str(R2I), str(R2I / "recall2imagine")]  # drop scripts/ from the import path

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import embodied  # noqa: E402
from recall2imagine import agent as agt  # noqa: E402
from recall2imagine import jaxutils  # noqa: E402
from recall2imagine import train  # noqa: E402


def patch(variant):
    """Replace sampling by the mode: for the action (2-d logits, [batch, actions]) unless
    `standard`, for the latent (3-d logits, [batch, stoch, classes]) only for `mode`."""
    original = jaxutils.OneHotDist.sample

    def sample(self, sample_shape=(), seed=None):
        logits = self.logits_parameter()
        latent = logits.ndim == 3
        if (latent and variant == "mode") or (not latent and variant != "standard"):
            mode = jax.nn.one_hot(jnp.argmax(logits, -1), logits.shape[-1], dtype=self.dtype)
            shape = (int(sample_shape),) if np.ndim(sample_shape) == 0 else tuple(sample_shape)  # tfd.Independent passes n
            return jnp.broadcast_to(mode, shape + mode.shape)
        return original(self, sample_shape, seed)

    jaxutils.OneHotDist.sample = sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir")
    parser.add_argument("--variant", choices=["standard", "sampled", "mode"], required=True)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--envs", type=int, default=32)
    args = parser.parse_args()
    logdir = pathlib.Path(args.logdir).resolve()
    patch(args.variant)
    import ruamel.yaml as yaml  # embodied.Config.load expects a pathlib-like read_text
    config = embodied.Config(yaml.YAML(typ="safe").load((logdir / "config.yaml").read_text()))
    config = config.update({"envs.amount": args.envs, "envs.parallel": "none", "envs.restart": False,
                            "jax.platform": "cpu", "jax.prealloc": False})
    env = train.make_envs(config)
    agent = agt.Agent(env.obs_space, env.act_space, embodied.Counter(), config)
    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(str(logdir / "checkpoint.ckpt"), keys=["agent"])
    scores = []
    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: scores.append(float(ep["reward"].astype(np.float64).sum())))
    driver(lambda *a: agent.policy(*a, mode="eval"), episodes=args.episodes)
    env.close()
    scores = np.array(scores[: args.episodes])
    print(json.dumps({"run": str(logdir.relative_to(ROOT)), "variant": args.variant, "episodes": len(scores),
                      "return": round(float(scores.mean()), 4),
                      "se": round(float(scores.std(ddof=1) / np.sqrt(len(scores))), 4)}), flush=True)


if __name__ == "__main__":
    main()
