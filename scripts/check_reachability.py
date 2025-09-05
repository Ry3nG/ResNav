from __future__ import annotations

import argparse
import numpy as np

from me5418_nav.envs import UnicycleNavEnv


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test: repeated resets must be reachable"
    )
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--num-pallets", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    env = UnicycleNavEnv({"render_mode": None})

    failures = 0
    resamples = []
    for k in range(int(args.n)):
        sd = int(rng.integers(0, 2**32 - 1))
        _, info = env.reset(
            seed=sd,
            options={
                "scenario": "blockage",
                "kwargs": {"num_pallets": int(args.num_pallets)},
            },
        )
        resamples.append(int(info.get("resample_count", 0)))

    print("[SMOKE] N=", int(args.n))
    print("[SMOKE] max_resample=", int(max(resamples) if resamples else 0))
    print("[SMOKE] mean_resample=", float(np.mean(resamples) if resamples else 0.0))

    env.close()


if __name__ == "__main__":
    main()
