import hydra
from omegaconf import DictConfig, OmegaConf, open_dict, read_write
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _save_config_with_mp(cfg: DictConfig, filename: str, output_dir: Path) -> None:
    logger.warning("This is a custom save_config function for multiprocessing in hydra.")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory {output_dir}: {e}")
    try:
        with open(str(output_dir / filename), "w", encoding="utf-8") as file:
            file.write(OmegaConf.to_yaml(cfg))
    except Exception as e:
        logger.error(f"Error writing config file to {output_dir / filename}: {e}")


hydra.core.utils._save_config = _save_config_with_mp
