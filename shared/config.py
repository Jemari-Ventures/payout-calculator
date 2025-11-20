"""
Configuration management with validation and caching.
"""
import json
import os
from typing import Optional, Dict, Any
import streamlit as st


class Config:
    """Configuration management with caching and validation."""

    CONFIG_FILE = "config.json"
    _cache: Optional[Dict[str, Any]] = None

    DEFAULT_CONFIG = {
        "data_source": {
            "type": "gsheet",
            "gsheet_url": "",
            "sheet_name": None,
            "postgres_table": "dispatcher",
            "postgres_query": None
        },
        "currency_symbol": "RM",
        "penalty_rate": 100.0,
    }

    @classmethod
    def load(cls, config_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file or create default.

        Args:
            config_file: Optional path to config file (defaults to CONFIG_FILE)

        Returns:
            Configuration dictionary
        """
        if cls._cache is not None:
            return cls._cache

        file_path = config_file or cls.CONFIG_FILE

        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    config = cls._merge_defaults(config)
                    cls._cache = config
                    return config
            except json.JSONDecodeError as e:
                if st:
                    st.error(f"Invalid JSON in config file: {e}")
                cls._cache = cls.DEFAULT_CONFIG.copy()
                return cls._cache
            except Exception as e:
                if st:
                    st.error(f"Error loading config: {e}")
                cls._cache = cls.DEFAULT_CONFIG.copy()
                return cls._cache
        else:
            # Create default config file
            cls.save(cls.DEFAULT_CONFIG, file_path)
            cls._cache = cls.DEFAULT_CONFIG.copy()
            return cls._cache

    @classmethod
    def _merge_defaults(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user config with defaults to ensure all keys exist."""
        merged = cls.DEFAULT_CONFIG.copy()
        merged.update(config)
        # Deep merge for nested dicts
        for key, value in config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = {**merged[key], **value}
        return merged

    @classmethod
    def save(cls, config: Dict[str, Any], config_file: Optional[str] = None) -> bool:
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary to save
            config_file: Optional path to config file

        Returns:
            True if successful, False otherwise
        """
        file_path = config_file or cls.CONFIG_FILE
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            cls._cache = config
            return True
        except Exception as e:
            if st:
                st.error(f"Error saving config: {e}")
            return False

    @classmethod
    def clear_cache(cls):
        """Clear the configuration cache."""
        cls._cache = None
