
import logging
import hydra
import rootutils
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from vae.model.vae import VAE


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")
log = logging.getLogger(__name__)

def train(cfg: DictConfig):
    log.info(cfg)
    return


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="train.yaml")
def main(cfg: DictConfig):
    return train(cfg)


if __name__ == "__main__":
    main()