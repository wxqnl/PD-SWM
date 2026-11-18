"""Evaluate the final agent of POPGym RL runs in the environment, with the latent sampled or at its mode.

dreamer.py evaluates the greedy actor but samples the posterior's stochastic state at every step
(collect_dyn_sample True, the DreamerV3 / R2I default), and the sample enters the recurrent state.
This script runs the final agent (latest_model.pt) exactly as dreamer.py's eval policy does, on the
same decks for every model (env seeds seed..seed+episodes-1), in two ways:

  sampled   the standard evaluation (posterior sampled), as in dreamer.py
  mode      the posterior's stochastic state at its mode (deterministic agent)

and prints the mean episode return and its standard error.

usage: python scripts/eval_popgym.py [--episodes 256] runs/popgym/popgym_RepeatPreviousEasy/<cell>/s0 ...
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[0] = str(ROOT)
sys.path.append(str(ROOT / "scripts" / "diag"))

import numpy as np
import torch

from diag_popgym_reward import load
from envs.popgym_env import PopGym


@torch.no_grad()
def rollout(config, wm, behavior, name, episodes, seed, sample):
    torch.manual_seed(0)
    envs = [PopGym(name, seed=seed + i) for i in range(episodes)]
    obs = [env.reset() for env in envs]
    latent = wm.dynamics.initial(episodes)
    action = torch.zeros(episodes, config.num_actions, device=config.device)
    ret, done = np.zeros(episodes), np.zeros(episodes, bool)
    while not done.all():
        batch = {k: np.stack([o[k] for o in obs]) for k in ("vector", "is_first", "is_last", "is_terminal")}
        data = wm.preprocess(batch)
        latent, _ = wm.dynamics.obs_step(latent, action, wm.encoder(data), data["is_first"], sample)
        action = behavior.actor(wm.dynamics.get_policy_feat(latent)).mode()
        while action.dim() > 2:  # spike steps
            action = action.mean(0)
        action = torch.nn.functional.one_hot(action.argmax(-1), config.num_actions).float()
        for i, env in enumerate(envs):
            if done[i]:
                continue
            obs[i], reward, done[i], _ = env.step(int(action[i].argmax()))
            ret[i] += reward
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for run in args.runs:
        logdir = pathlib.Path(run).resolve()
        config, wm, behavior = load(logdir, args.device)
        name = config.task.split("_", 1)[1]  # popgym_RepeatPreviousEasy -> RepeatPreviousEasy
        out = {"run": str(logdir.relative_to(ROOT)), "task": name, "episodes": args.episodes, "seed": args.seed}
        for mode, sample in (("sampled", True), ("mode", False)):
            ret = rollout(config, wm, behavior, name, args.episodes, args.seed, sample)
            out[mode] = round(float(ret.mean()), 4)
            out[mode + "_se"] = round(float(ret.std(ddof=1) / np.sqrt(len(ret))), 4)
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
