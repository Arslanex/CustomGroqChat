"""
Custom exceptions for 'CustomGroqChat' module.

This file defines custom exceptions that can be raised during the execution of the module.
Every exception inherits from the built-in Exception class, and each exception has a custom message.
These exceptions can be used to handle specific error cases in the module's functionality.
"""

from typing import Any, Dict, Optional


class CustomGroqChatException(Exception):
    """Base class for all exceptions raised by the CustomGroqChat module."""

    def __init__(self, message: str, error_code: Optional[str] = None) -> None:
        """
        Initialize the exception with a message and optional error code.

        Args:
            message (str): The error message.
            error_code (Optional[int]): An optional error code associated with the exception.
        """
        super().__init__(message)                                                                                       # Call the base class constructor with the message
        self.message = message                                                                                          # Store the message
        self.error_code = error_code                                                                                    # Store the error code


    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary containing the message and error code.
        """
        return {                                                                                                        # Convert the exception to a dictionary
            "message": self.message,                                                                                    # Include the message in the dictionary
            "error_code": self.error_code,                                                                              # Include the error code in the dictionary
            "error_type": self.__class__.__name__                                                                       # Include the type of the exception
        }


class ConfigLoaderException(CustomGroqChatException):
    """Exception raised for errors in the configuration loading process."""

    ERROR_CODE = "CONFIG_LOADER_ERROR"                                                                                  # Error code for configuration loading errors

    def __init__(self, message: str, config_key: Optional[str] = None) -> None:
        """
        Initialize the exception with a message and optional configuration key.

        Args:
            message (str): The error message.
            config_key (Optional[str]): An optional configuration key associated with the exception.
        """
        super().__init__(message, error_code=self.ERROR_CODE)                                                           # Call the base class constructor with the message and error code
        self.config_key = config_key                                                                                    # Store the configuration key


    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary containing the message, error code, and configuration key.
        """
        error_dict = super().to_dict()                                                                                  # Get the base dictionary representation

        if self.config_key:                                                                                             # If a configuration key is provided
            error_dict["config_key"] = self.config_key                                                                  # Add the configuration key to the dictionary

        return error_dict                                                                                               # Return the complete dictionary representation                                                                                    # Return the complete dictionary representation"""