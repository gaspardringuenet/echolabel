"""Configuration management for Echolabel"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import yaml

# Python 3.9+
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files


def get_config_dir() -> Path:
    """Get the user configuration directory"""
    if os.name == "posix":
        # macOs/Linux: ~/.config/echolabel
        config_home = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
        return Path(config_home) / "echolabel"
    else:
        # windows: %APPDATA%/echolabel
        return Path(os.environ["APPDATA"]) / "echolabel"


def get_user_config_path() -> Path:
    """Get path to user's config file"""
    return get_config_dir() / "config.yaml"


def init_user_config() -> Path:
    """Initialize user config directory ad copy default config if needed"""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    user_config = get_user_config_path()

    if not user_config.exists():
        # Copy default config from package
        default_config = files("echolabel").joinpath("config/config.default.yaml")
        with default_config.open("r") as src:
            with open(user_config, "w") as dst:
                dst.write(src.read())

    return user_config


@dataclass
class Config:
    "Default configuration for echolabel app"

    channels_frequency_nominal: List[float]
    frame_width: int
    range_samples_slice: slice
    vmin: float
    vmax: float
    echogram_cmap: str
    reuse_images: bool

    def update(self, overrides: dict) -> None:
        """Update configuration with overrides dictionary"""
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """
        Load configuration with fallback chain:
        1. Specified config file
        2. User config (~/.config.echolabel/config.yaml)
        3. Package defaults
        """
        if config_path:
            return cls.from_file(config_path)

        user_config = get_user_config_path()
        if user_config.exists():
            return cls.from_file(user_config)

        return cls.from_defaults()

    @classmethod
    def from_defaults(cls) -> "Config":
        """Load default configuration from packages YAML file"""
        config_file = files("echolabel").joinpath("config/config.default.yaml")
        with config_file.open("r") as f:
            data = yaml.safe_load(f)
        _format_depth_range_slice(data)
        return cls(**data)

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        """Load configuration from YAML file"""
        path = Path(path)

        with open(path, "r") as f:
            if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        config = cls.from_defaults()
        _format_depth_range_slice(data)
        config.update(data)
        return config

    def to_file(self, path: Path) -> None:
        """Save configuration to YAML file"""
        path = Path(path)
        data = asdict(self)

        if isinstance(data.get("range_samples_slice"), slice):
            s = data["range_samples_slice"]
            data["range_samples_slice"] = {"start": s.start, "stop": s.stop, "step": s.step}

        with open(path, "w") as f:
            if path.suffix in [".yaml", ".yml"]:
                yaml.dump(data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported config export file format: {path.suffix}")


def _format_depth_range_slice(data: dict, key: str = "range_samples_slice") -> None:
    """Inplace formatting of the depth rangle samples slice from dict to slice"""
    if key in data and isinstance(data[key], dict):
        s = data[key]
        data[key] = slice(s.get("start"), s.get("stop"), s.get("step"))
