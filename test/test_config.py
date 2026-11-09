from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import echolabel.config.config as config_module
from echolabel.config.config import Config, get_config_dir, init_user_config


def test_config_from_defaults():
    """Load default configuration"""
    config = Config.from_defaults()
    assert isinstance(config, Config)
    assert config.frame_width > 0
    assert config.vmin < config.vmax


def test_config_from_file(tmp_path, default_config):
    """Load config from a YAML file"""
    config_file = tmp_path / "test_config.yaml"
    default_config.vmin = -100.0  # Modify
    default_config.to_file(config_file)

    loaded = Config.from_file(config_file)
    assert loaded.vmin == -100.0


def test_config_update():
    """Update config with overrides"""
    config = Config.from_defaults()
    original_vmax = config.vmax

    config.update({"vmin": -999.0})
    assert config.vmin == -999.0
    # Other values unchanged
    assert config.vmax == original_vmax


def test_config_to_file(tmp_path, default_config):
    """Save config to YAML file"""
    config_file = tmp_path / "output.yaml"
    default_config.to_file(config_file)

    assert config_file.exists()
    loaded = Config.from_file(config_file)
    assert loaded.frame_width == default_config.frame_width


def test_oklab_config_defaults_and_partial_override(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"oklab": {"theta": 1.0}}))
    config = Config.from_file(config_file)
    assert config.oklab == {
        "sat_max": 28.3,
        "l_range": [0.3, 0.75],
        "c_max": 0.25,
        "theta": 1.0,
    }

    config.to_file(config_file)
    assert Config.from_file(config_file).oklab == config.oklab
    assert Config.from_defaults().oklab == {
        "sat_max": 28.3,
        "l_range": [0.3, 0.75],
        "c_max": 0.25,
        "theta": 2.67,
    }
    config_file.write_text("vmin: -80\n")
    assert Config.from_file(config_file).oklab == Config.from_defaults().oklab


@pytest.mark.parametrize(
    "settings",
    [
        {"sat_max": 0},
        {"sat_max": float("inf")},
        {"c_max": -1},
        {"theta": float("nan")},
        {"l_range": [0.8, 0.2]},
        {"l_range": [-0.1, 0.5]},
        {"l_range": [0.5]},
        {"theta": "bad"},
        {"unknown": 1},
    ],
)
def test_invalid_oklab_config(settings):
    with pytest.raises(ValueError, match="oklab|OKLab"):
        Config.from_defaults().update({"oklab": settings})


def test_oklab_config_must_be_a_mapping(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("oklab: invalid\n")
    with pytest.raises(ValueError, match="oklab must be a mapping"):
        Config.from_file(config_file)


def test_config_load_fallback(tmp_path, monkeypatch):
    """Test Config.load() fallback chain"""
    # Create a user config
    user_config = tmp_path / "user_config.yaml"
    config = Config.from_defaults()
    config.vmin = -75.0
    config.to_file(user_config)

    # Load with fallback to specific file
    loaded = Config.load(config_path=user_config)
    assert loaded.vmin == -75.0


def test_get_config_dir_posix(monkeypatch):
    """Test config directory on POSIX systems"""
    monkeypatch.setattr(config_module, "os", SimpleNamespace(name="posix", environ={}))
    assert get_config_dir() == Path.home() / ".config" / "echolabel"


def test_get_config_dir_posix_xdg(monkeypatch, tmp_path):
    monkeypatch.setattr(config_module, "os", SimpleNamespace(name="posix", environ={"XDG_CONFIG_HOME": str(tmp_path)}))
    assert get_config_dir() == tmp_path / "echolabel"


def test_get_config_dir_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(config_module, "os", SimpleNamespace(name="nt", environ={"APPDATA": str(tmp_path)}))
    assert get_config_dir() == tmp_path / "echolabel"


def test_init_user_config(tmp_path, monkeypatch):
    """Test user config initialization"""
    mock_config_dir = tmp_path / "config"

    monkeypatch.setattr("echolabel.config.config.get_config_dir", lambda: mock_config_dir)

    result = init_user_config()
    assert result.parent == mock_config_dir
    assert result.exists()
    assert result.name == "config.yaml"
