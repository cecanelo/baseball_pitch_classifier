import yaml
import os


def load_config(config_path: str) -> dict:
    """
    Load and parse a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file is not valid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse config file {config_path}: {e}")
