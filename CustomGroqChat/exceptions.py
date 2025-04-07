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

        return error_dict                                                                                               # Return the complete dictionary representation

class RateLimitExceededException(CustomGroqChatException):
    """Exception raised when the rate limit is exceeded."""

    ERROR_CODE = "RATE_LIMIT_EXCEEDED"                                                                                  # Error code for rate limit exceeded errors

    def __init__(self,
                 message: str,
                 limit_type: str,
                 current_value: int,
                 limit_value: int,
                 time_period: str
                 ) -> None:
        """
        Initialize the exception with a message.

        Args:
            message (str): The error message.
            limit_type (str): The type of rate limit exceeded.
            current_value (int): The current value of the rate limit.
            limit_value (int): The maximum value of the rate limit.
            time_period (str): The time period for the rate limit.
        """
        super().__init__(message, error_code=self.ERROR_CODE)                                                           # Call the base class constructor with the message and error code
        self.message = message                                                                                          # Store the message
        self.limit_type = limit_type                                                                                    # Store the type of rate limit
        self.current_value = current_value                                                                              # Store the current value of the rate limit
        self.limit_value = limit_value                                                                                  # Store the maximum value of the rate limit
        self.time_period = time_period                                                                                  # Store the time period for the rate limit

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary containing the message, error code, and rate limit details.
        """
        error_dict = super().to_dict()                                                                                  # Get the base dictionary representation

        error_dict.update({                                                                                             # Update the dictionary with rate limit details
            "limit_type": self.limit_type,                                                                              # Include the type of rate limit in the dictionary
            "current_value": self.current_value,                                                                        # Include the current value of the rate limit in the dictionary
            "limit_value": self.limit_value,                                                                            # Include the maximum value of the rate limit in the dictionary
            "time_period": self.time_period                                                                             # Include the time period for the rate limit in the dictionary
        })

        return error_dict                                                                                               # Return the complete dictionary representation

class APICallException(CustomGroqChatException):
    """Exception raised for errors during API calls."""

    ERROR_CODE = "API_CALL_ERROR"                                                                                       # Error code for API call errors

    def __init__(self,
                 message: str,
                 status_code: Optional[int] = None,
                 reponse_body: Optional[Dict[str, Any]] = None,
                 ) -> None:
        """
        Initialize the exception with a message.

        Args:
            - message (str): The error message.
            - status_code (Optional[int]): The HTTP status code associated with the error.
            - response_body (Optional[Dict[str, Any]]): The response body associated with the error.
        """

        error_code = f'{self.ERROR_CODE}_{status_code}' if status_code else self.ERROR_CODE                             # Generate error code based on status code
        super().__init__(message, error_code=error_code)                                                                # Call the base class constructor with the message and error code
        self.status_code = status_code                                                                                  # Store the HTTP status code
        self.response_body = reponse_body                                                                               # Store the response body

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary containing the message, error code, and API call details.
        """
        error_dict = super().to_dict()                                                                                  # Get the base dictionary representation

        if self.status_code:                                                                                            # If a status code is provided
            error_dict["status_code"] = self.status_code                                                                # Add the status code to the dictionary

        if self.response_body:                                                                                          # If a response body is provided
            error_dict["response_body"] = self.response_body                                                            # Add the response body to the dictionary

        return error_dict                                                                                               # Return the complete dictionary representation

class ModelNotFoundException(CustomGroqChatException):
    """Exception raised when a model is not found."""

    ERROR_CODE = "MODEL_NOT_FOUND"                                                                                     # Error code for model not found errors

    def __init__(self, message: str, model_name: Optional[str] = None) -> None:
        """
        Initialize the exception with a message and optional model name.

        Args:
            message (str): The error message.
            model_name (Optional[str]): The name of the model that was not found.
        """
        super().__init__(message, error_code=self.ERROR_CODE)                                                           # Call the base class constructor with the message and error code
        self.model_name = model_name                                                                                    # Store the name of the model

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary representation

        Returns:
            Dict[str, Any]: A dictionary containing the message, error code, and API call details.
        """

        error_dict = super().to_dict()

        if self.model_name:                                                                                             # If a model name is provided
            error_dict["model_name"] = self.model_name                                                                  # Add the model name to the dictionary

        return error_dict

class TokenLimitExceededException(CustomGroqChatException):
    """Exception raised when the token limit is exceeded."""

    ERROR_CODE = "TOKEN_LIMIT_EXCEEDED"                                                                                # Error code for token limit exceeded errors

    def __init__(self, message: str, token_count: int, token_limit: int, model_name:str) -> None:
        """
        Initialize the exception with a message and token limit.

        Args:
            message (str): The error message.
            token_limit (int): The maximum token limit.
        """
        super().__init__(message, error_code=self.ERROR_CODE)                                                           # Call the base class constructor with the message and error code
        self.token_count = token_count                                                                                  # Store the current token count
        self.token_limit = token_limit                                                                                  # Store the maximum token limit
        self.model_name = model_name                                                                                    # Store the name of the model

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary containing the message, error code, and API call details.
        """
        error_dict = super().to_dict()                                                                                  # Get the base dictionary representation

        error_dict.update({                                                                                             # Update the dictionary with token limit details
            "token_count": self.token_count,                                                                            # Include the current token count in the dictionary
            "token_limit": self.token_limit,                                                                            # Include the maximum token limit in the dictionary
            "model_name": self.model_name                                                                               # Include the name of the model in the dictionary
        })

        return error_dict                                                                                               # Return the complete dictionary representation