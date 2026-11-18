"""Memory Maze offline dataset (Pasukonis et al., ICLR 2023) for the probing benchmark: trajectories, TBTT streams
and the constant baseline, numpy only (shared by probe_mm.py and probe_mm_r2i.py, which runs in R2I's JAX
environment)."""
import concurrent.futures
import io
import pathlib
import threading
import zipfile

import numpy as np

KEYS = ("image", "action", "reward", "targets_vec", "maze_layout", "agent_pos", "agent_dir")


class Episodes:
    """Trajectories of an offline Memory Maze dataset: npz files inside zip archives (the official
    release) or in a directory."""

    def __init__(self, sources):
        self.items = []
        for src in sources:
            src = pathlib.Path(src)
            if src.suffix == ".zip":
                with zipfile.ZipFile(src) as zf:
                    self.items += [(src, n) for n in sorted(zf.namelist()) if n.endswith(".npz")]
            else:
                self.items += [(None, f) for f in sorted(src.glob("*.npz"))]
        self._local = threading.local()  # one ZipFile handle per thread

    def __len__(self):
        return len(self.items)

    def load(self, i):
        archive, member = self.items[i]
        if archive is None:
            raw = np.load(member)
        else:
            handles = self._local.__dict__.setdefault("zips", {})
            if archive not in handles:
                handles[archive] = zipfile.ZipFile(archive)
            raw = np.load(io.BytesIO(handles[archive].read(member)))
        ep = {k: raw[k] for k in KEYS}
        if ep["maze_layout"].ndim == 3:  # constant within the trajectory
            ep["maze_layout"] = ep["maze_layout"][0]
        ep["is_first"] = np.zeros(len(ep["reward"]), bool)
        ep["is_first"][0] = True
        return ep


def split_sources(data):
    data = pathlib.Path(data)
    train = sorted(data.glob("train*.zip")) or [data / "train"]
    evals = sorted(data.glob("eval*.zip")) or [data / "eval"]
    return train, evals


class Streams:
    """B streams of consecutive chunks through random training trajectories (TBTT). Each stream
    starts at a random offset of its first trajectory so that episode starts are not synchronised."""

    def __init__(self, episodes, batch, length, seed, workers=8):
        self.eps, self.B, self.L = episodes, batch, length
        self.rng = np.random.RandomState(seed)
        self.pool = concurrent.futures.ThreadPoolExecutor(workers)
        self.cur = [f.result() for f in [self._prefetch() for _ in range(batch)]]
        self.next_ep = [self._prefetch() for _ in range(batch)]
        self.pos = [self.rng.randint(0, len(ep["reward"]) - length) for ep in self.cur]

    def _prefetch(self):
        return self.pool.submit(self.eps.load, self.rng.randint(len(self.eps)))

    def next(self):
        for b in range(self.B):
            if self.pos[b] + self.L > len(self.cur[b]["reward"]):
                self.cur[b], self.pos[b] = self.next_ep[b].result(), 0
                self.next_ep[b] = self._prefetch()
        batch = {k: np.stack([ep[k][p: p + self.L] for ep, p in zip(self.cur, self.pos)])
                 for k in (*KEYS, "is_first") if k != "maze_layout"}
        batch["maze_layout"] = np.stack([ep["maze_layout"] for ep in self.cur])
        self.pos = [p + self.L for p in self.pos]
        return batch


def constant_baseline(train, evals, idx, pool, n=500):
    """Training-set mean prediction, scored like the models (second half of the evaluation data)."""
    tr = list(pool.map(train.load, range(0, len(train), max(1, len(train) // n))))
    mean_vec = np.concatenate([ep["targets_vec"][1:] for ep in tr]).reshape(-1, 3, 2).mean(0)
    wall_freq = np.stack([ep["maze_layout"] for ep in tr]).reshape(-1, 81).mean(0)
    ev = list(pool.map(evals.load, idx))
    vec = np.stack([ep["targets_vec"][501:] for ep in ev]).reshape(-1, 3, 2)
    walls = np.stack([ep["maze_layout"] for ep in ev]).reshape(-1, 81)
    return {"const_objects_mse": float(((vec - mean_vec) ** 2).sum(-1).mean()),
            "const_walls_acc": float(((wall_freq > 0.5) == walls).mean())}
