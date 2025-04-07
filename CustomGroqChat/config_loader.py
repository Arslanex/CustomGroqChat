"""
ConfigLoader class for loading configuration of GROQ CLOUD API client.

This class is responsible for loading the configuration from a JSON file,
validating it, and providing access to the configuration parameters.
"""

import os
import json
from typing import Any, Dict, List, Optional

from .exceptions import ConfigLoaderException


class ConfigLoader:
    """Class for loading configuration of GROQ CLOUD API client."""

    def __init__(self, config_path: str) -> None:
        """
        Initialize the ConfigLoader with a configuration file path.

        Args:
            config_path (str): Path to the configuration file.

        Raises:
            ConfigLoaderException: If the configuration file path is invalid.
        """
        if not config_path or not isinstance(config_path, str):                                                         # Check if the config_path is a valid string
            raise ConfigLoaderException(                                                                                # Raise an exception if the config_path is invalid
                "Invalid configuration file path provided.",                                                   # Error message associated with the exception
                config_key="config_path"                                                                                # Configuration key associated with the error
            )

        self.config_path = config_path                                                                                  # Store the configuration file path
        self.config: Optional[Dict[str, Any]] = {}                                                                      # Initialize config as an empty dictionary


    def load_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the configuration from the specified file.

        Returns:
            Dictionary containing the configuration parameters.

        Raises:
            ConfigLoaderException: If there is an error loading the configuration.
        """
        if not os.path.exists(self.config_path):                                                                        # Check if the configuration file exists
            raise ConfigLoaderException(                                                                                # Raise an exception if the config file does not exist
                "Configuration file does not exist.",                                                          # Error message associated with the exception
                config_key="config_path"                                                                                # Configuration key associated with the error
            )

        try:
            with open(self.config_path, "r") as config_file:                                                            # Open the configuration file for reading
                self.config = json.load(config_file)                                                                    # Load the JSON content into a dictionary
        except json.decoder.JSONDecodeError:                                                                            # Handle JSON decoding errors
            raise ConfigLoaderException(                                                                                # Raise an exception if the JSON content is invalid
                "Configuration file could not be decoded.",                                                    # Error message associated with the exception
                config_key="file_format"                                                                                # Configuration key associated with the error
            )
        except Exception as e:                                                                                          # Handle any other exceptions
            raise ConfigLoaderException(                                                                                # Raise an exception for any other errors
                f"An error occurred while loading the configuration: {str(e)}",                                # Error message associated with the exception
                config_key="file_format"                                                                                # Configuration key associated with the error
            )

        self._validate_config()                                                                                         # Validate the loaded configuration
        return self.config                                                                                              # Return the loaded configuration dictionary


    def _validate_config(self) -> None:
        """
        Validate the loaded configuration to ensure it contains required keys are present.

        Rate Limit Values: req_per_minute, req_per_day, token_per_minute, token_per_day
        (values can be either positive integers or '-1' for unlimited)

        Raises:
            ConfigLoaderException: If the configuration is invalid.
        """
        required_fields = [
            "base_url",
            "api_key",
            "req_per_minute",
            "req_per_day",
            "token_per_minute",
            "token_per_day",
        ]

        if not self.config:
            raise ConfigLoaderException(                                                                                # Raise an exception if the config is empty
                "Empty configuration file or no models defined",                                               # Error message associated with the exception
                config_key="model_values"                                                                               # Configuration key associated with the error
            )

        for model, config in self.config.items():                                                                       # Iterate through each model in the configuration
            # Check missing required fields                                                                             # Check for missing required fields
            missing_fields = [field for field in required_fields if field not in config]                                # Identify missing fields

            if missing_fields:                                                                                          # If there are missing fields
                raise ConfigLoaderException(                                                                            # Raise an exception for missing fields
                    message=f"The following {model} requires configuration fields: {', '.join(missing_fields)}",        # Error message associated with the exception
                    config_key=f"{model}.{missing_fields[0] if missing_fields else 'fields'}"                           # Provide the first missing field as the config key,
                )

            # Validate rate limit values
            rate_limit_fields = ["req_per_minute", "req_per_day", "token_per_minute", "token_per_day"]                  # Define rate limit fields
            for field in rate_limit_fields:                                                                             # Iterate through each rate limit field
                value = config.get(field)                                                                               # Get the value of the field from the configuration
                if not isinstance(value, int) or (value < 0 and value != -1):                                  # Check if the value is a positive integer or -1 for unlimited
                    raise ConfigLoaderException(                                                                        # Raise an exception for invalid rate limit values
                        message=f"Invalid value for {field} in {model}: {value}. Must be a positive integer "           # Error message associated with the exception
                                f"or -1 for unlimited.",
                        config_key=f"{model}.{field}"  # Provide the specific field as the config key                   # Configuration key associated with the error
                    )


    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the configuration for a specific model.

        Args:
            model_name (str): Name of the model.

        Returns:
            Dictionary containing the model configuration, or None if not found.
        """
        return self.config.get(model_name) if self.config else None                                                      # Return the configuration for the specified model"""