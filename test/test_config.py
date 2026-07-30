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
    monkeypatch.setattr("os.name", "posix")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    config_dir = get_config_dir()
    assert ".config/echolabel" in str(config_dir)


def test_init_user_config(tmp_path, monkeypatch):
    """Test user config initialization"""
    mock_config_dir = tmp_path / "config"

    monkeypatch.setattr("echolabel.config.config.get_config_dir", lambda: mock_config_dir)

    result = init_user_config()
    assert result.parent == mock_config_dir
    assert result.exists()
    assert result.name == "config.yaml"
