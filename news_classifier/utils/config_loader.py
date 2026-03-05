"""
Configuration loader utility
"""


def load_config():
    """Load configuration from config.py"""
    try:
        from config import CONFIG, STOPWORDS
        return CONFIG, STOPWORDS
    except ImportError as e:
        print(f"Error loading config: {e}")
        raise


def get_config_value(config, *keys):
    """Safely get nested config value"""
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value