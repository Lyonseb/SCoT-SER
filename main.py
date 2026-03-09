import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from src.train_ddp import train_ddp
from src.train_deepspeed import train_deepspeed
from config.config import Config
import time
import argparse
import sys

# Register structured config so Hydra knows keys like `train.*`, `data.*`, etc.
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
def parse_deepspeed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args

@hydra.main(config_name="config", version_base=None)
def main_hydra(cfg):
    run_config = Config()
    run_config.env.save_path += f"/{time.strftime('%H-%M-%S')}"
    cfg = OmegaConf.merge(run_config, cfg)
    if cfg.train.train_strategy == "ddp":
        train_ddp(cfg)
    else:
        train_deepspeed(cfg)

if __name__ == "__main__":
    # Make Hydra and DeepSpeed CLI arguments compatible
    deepspeed_args, remaining_args = parse_deepspeed_args()
    sys.argv = [sys.argv[0]] + remaining_args  # Pass only arguments that Hydra can parse
    main_hydra()
