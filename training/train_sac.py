from __future__ import annotations

import hydra
from omegaconf import DictConfig

from training.common import train_with_algo


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    train_with_algo(cfg, algo_name="sac")


if __name__ == "__main__":
    main()
