"""
Configuration module.
====================
Application configuration and constants.
"""

# Application settings
APP_NAME = "Mini Repository Example"
VERSION = "1.0.0"
DEBUG = False

# Data processing settings
MAX_BATCH_SIZE = 100
DEFAULT_TIMEOUT = 30

# Validation settings
REQUIRED_FIELDS = ["id", "value", "type"]
VALID_TYPES = ["A", "B", "C"]

# Output settings
OUTPUT_FORMAT = "json"
INCLUDE_TIMESTAMP = True

# Database settings (not actually used, just for demonstration)
DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "mini_app",
    "username": "user",
    "password": "password",
}


def get_config() -> dict:
    """Get the current configuration as a dictionary."""
    return {
        "app_name": APP_NAME,
        "version": VERSION,
        "debug": DEBUG,
        "max_batch_size": MAX_BATCH_SIZE,
        "default_timeout": DEFAULT_TIMEOUT,
        "required_fields": REQUIRED_FIELDS,
        "valid_types": VALID_TYPES,
        "output_format": OUTPUT_FORMAT,
        "include_timestamp": INCLUDE_TIMESTAMP,
    }


