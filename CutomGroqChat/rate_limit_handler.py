"""
RateLimitHandler class to handle rate limiting for GROQ CLOUD API client requests.

This class is responsible for managing the rate limit, including checking if the limit has been exceeded,
waiting for the appropriate time to retry, and handling the rate limit exceeded exception.
"""

import time
from typing import Dict, Any, List, Tuple, Optional

from .exceptions import RateLimitExceededException


class RateLimitHandler:
    """Class to handle rate limiting for GROQ CLOUD API client requests."""
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RateLimitHandler with configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing rate limit settings.
        """
        # Retrieve rate limit settings from the configuration
        self.req_per_min = config.get("req_per_min", 15)                                                                # Set the maximum number of requests per minute (deault: 15 requests/min [GROQ CLOUD API - 04/2025]
        self.req_per_day = config.get("req_per_day", 1000)                                                              # Set the maximum number of requests per day (default: 1000 requests/day [GROQ CLOUD API - 04/2025]
        self.tokens_per_min = config.get("tokens_per_min", 6000)                                                        # Set the maximum number of tokens per minute (default: 6000 tokens/min [GROQ CLOUD API - 04/2025]
        self.tokens_per_day = config.get("tokens_per_day", 250000)                                                      # Set the maximum number of tokens per day (default: 250000 tokens/day [GROQ CLOUD API - 04/2025]

        # Initialize rate limit counters for tracking usage
        self.req_minute_counter = 0                                                                                     # Counter for requests in the current minute
        self.req_day_counter = 0                                                                                        # Counter for requests in the current day
        self.tokens_minute_counter = 0                                                                                  # Counter for tokens in the current minute
        self.tokens_day_counter = 0                                                                                     # Counter for tokens in the current day

        # Timestamps for last reset of rate limit counters
        self._last_minute_reset = time.time()                                                                           # Timestamp for the last minute reset
        self._last_day_reset = time.time()                                                                              # Timestamp for the last day reset

    def _reset_minute_counters(self) -> None:
        """
        Reset the minute counter and update the last reset timestamp.
        """
        current_time = time.time()                                                                                      # Get the current time
        if current_time - self._last_minute_reset >= 60:                                                                # If a minute has passed since the last reset
            self.req_minute_counter = 0                                                                                 # Reset the request counter for the current minute
            self.tokens_minute_counter = 0                                                                              # Reset the token counter for the current minute
            self._last_minute_reset = current_time                                                                      # Update the last minute reset timestamp

    def _reset_day_counters(self) -> None:
        """
        Reset the day counter and update the last reset timestamp.
        """
        current_time = time.time()                                                                                      # Get the current time
        if current_time - self._last_day_reset >= 86400:    # 86400 seconds = 24 hours                                  # If a day has passed since the last reset
            self.req_day_counter = 0                                                                                    # Reset the request counter for the current day
            self.tokens_day_counter = 0                                                                                 # Reset the token counter for the current day
            self._last_day_reset = current_time                                                                         # Update the last day reset timestamp

    def _reset_counters(self) -> None:
        """
        Reset both minute and day counters.
        """
        self._reset_minute_counters()                                                                                   # Reset the minute counters
        self._reset_day_counters()                                                                                      # Reset the day counters

    def can_make_request(self, token_count: int) -> Tuple[bool, List[str]]:
        """
        Check if a request can be made based on the current rate limits.

        Args:
            token_count (int): The number of tokens required for the request.

        Returns:
            Tuple containing:
                - bool: True if the request can be made, False otherwise.
                - List[str]: A list of reasons why the request cannot be made.

        Raises:
            TypeError: If token_count is not an integer.
        """
        if not isinstance(token_count, int):                                                                            # Check if token_count is an integer
            raise TypeError("token_count must be an integer")                                                           # Raise TypeError if token_count is not an integer

        # Reset the rate limit counters if necessary
        self._reset_counters()                                                                                          # Reset the rate limit counters

        reasons = []                                                                                                    # Initialize an empty list to store reasons for rate limit exceeded
        # Check all the rate limits, treating '-1' as unlimited
        # Check if the request per minute limit is exceeded
        if self.req_per_min != -1 and self.req_minute_counter >= self.req_per_min:                                      # Check if the request per minute limit is exceeded
            reasons.append(f"Rate limit exceeded - Minute Request Limit")                                               # Add reason to the list

        if self.tokens_per_min != -1 and self.tokens_minute_counter + token_count > self.tokens_per_min:                # Check if the tokens per minute limit is exceeded
            reasons.append(f"Rate limit exceeded - Minute Token Limit")                                                 # Add reason to the list

        # Check if the request per day limit is exceeded
        if self.req_per_day != -1 and self.req_day_counter >= self.req_per_day:                                         # Check if the request per day limit is exceeded
            reasons.append(f"Rate limit exceeded - Daily Request Limit")                                                # Add reason to the list

        if self.tokens_per_day != -1 and self.tokens_day_counter + token_count > self.tokens_per_day:                   # Check if the tokens per day limit is exceeded
            reasons.append(f"Rate limit exceeded - Daily Token Limit")                                                  # Add reason to the list

        return len(reasons) == 0, reasons                                                                               # Return True if no reasons are found, otherwise return False and the list of reasons

    def check_request(self, token_count: int, strictly: bool = False) -> None:
        """
        Check if a request can be made based on the current rate limits and wait if necessary.
        If request can't be made, raises RateLimitExceededException.

        Args:
            token_count (int): The number of tokens required for the request.
            strictly (bool): If True, raises RateLimitExceededException if the request can't be made.

        Raises:
            RateLimitExceededException: If the request can't be made and strictly is True.
            TypeError: If token_count is not an integer.
        """
        if not isinstance(token_count, int):                                                                            # Check if token_count is an integer
            raise TypeError("token_count must be an integer")                                                           # Raise TypeError if token_count is not an integer

        can_process, reasons = self.can_make_request(token_count)                                                       # Check if the request can be made

        if not can_process and strictly:                                                                                # If the request can't be made and strictly is True
            # Find the specific reason for the rate limit exceeded
            for reason in reasons:                                                                                      # Iterate through the reasons
                if "Minute Request Limit" in reason:                                                                    # If the reason is related to the minute request limit
                    raise RateLimitExceededException(                                                                   # Raise RateLimitExceededException
                        message=f"Rate limit exceeded - Minute Request Limit",                                          # Set the message to indicate the reason
                        limit_type="request",                                                                           # Set the limit type to request
                        current_value=self.req_minute_counter,                                                          # Set the current value to the minute request counter
                        limit_value=self.req_per_min,                                                                   # Set the limit value to the minute request limit
                        time_period="minute"                                                                            # Set the time period to minute
                    )

                elif "Minute Token Limit" in reason:                                                                    # If the reason is related to the minute token limit
                    raise RateLimitExceededException(                                                                   # Raise RateLimitExceededException
                        message=f"Rate limit exceeded - Minute Token Limit",                                            # Set the message to indicate the reason
                        limit_type="token",                                                                             # Set the limit type to token
                        current_value=self.tokens_minute_counter,                                                       # Set the current value to the minute token counter
                        limit_value=self.tokens_per_min,                                                                # Set the limit value to the minute token limit
                        time_period="minute"                                                                            # Set the time period to minute
                    )

                elif "Daily Request Limit" in reason:                                                                   # If the reason is related to the daily request limit
                    raise RateLimitExceededException(                                                                   # Raise RateLimitExceededException
                        message=f"Rate limit exceeded - Daily Request Limit",                                           # Set the message to indicate the reason
                        limit_type="request",                                                                           # Set the limit type to request
                        current_value=self.req_day_counter,                                                             # Set the current value to the daily request counter
                        limit_value=self.req_per_day,                                                                   # Set the limit value to the daily request limit
                        time_period="day"                                                                               # Set the time period to day
                    )

                elif "Daily Token Limit" in reason:                                                                     # If the reason is related to the daily token limit
                    raise RateLimitExceededException(                                                                   # Raise RateLimitExceededException
                        message=f"Rate limit exceeded - Daily Token Limit",                                             # Set the message to indicate the reason
                        limit_type="token",                                                                             # Set the limit type to token
                        current_value=self.tokens_day_counter,                                                          # Set the current value to the daily token counter
                        limit_value=self.tokens_per_day,                                                                # Set the limit value to the daily token limit
                        time_period="day"                                                                               # Set the time period to day
                    )

            # If we get here, we have reasons but don't have a specific handler for them
            if reasons:  # Only raise if there are actually reasons
                raise RateLimitExceededException(                                                                       # Raise RateLimitExceededException
                    message=": ".join(reasons),                                                                         # Set the message to indicate the reasons
                    limit_type="unknown",                                                                               # Set the limit type to unknown
                    current_value=0,                                                                                    # Set the current value to 0
                    limit_value=0,                                                                                      # Set the limit value to 0
                    time_period="unknown"                                                                               # Set the time period to unknown
                )

    def update_counters(self, token_count: int) -> None:
        """
        Update the rate limit counters after a request is made.

        Args:
            token_count (int): The number of tokens used in the request.
        """
        self.tokens_minute_counter += token_count                                                                       # Increment the token counter for the current minute
        self.tokens_day_counter += token_count                                                                          # Increment the token counter for the current day
        self.req_minute_counter += 1                                                                                    # Increment the request counter for the current minute
        self.req_day_counter += 1                                                                                       # Increment the request counter for the current day

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the rate limit counters.

        Returns:
            Dict[str, Any]: A dictionary containing the current status of the rate limit counters.
        """
        # Reset the rate limit counters if necessary
        self._reset_minute_counters()                                                                                   # Reset the minute counters

        # Helper function to format limits, -1 indicates unlimited
        def format_limit(limit: int) -> str:                                                                            # Helper function to format limits
            return "Unlimited" if limit == -1 else limit                                                                # Return "Unlimited" if the limit is -1, otherwise return the limit

        return {                                                                                                        # Return a dictionary containing the current status of the rate limit counters
            "requests":{                                                                                                # Dictionary for request limits
                "minute": {                                                                                             # Dictionary for minute request limits
                    "current": self.req_minute_counter,                                                                 # Current request counter for the minute
                    "limit": format_limit(self.req_per_min),                                                            # Maximum request limit for the minute
                    "display": f"{self.req_minute_counter}/{format_limit(self.req_per_min)}",                           # Display string for the current and maximum request limits
                },
                "day": {                                                                                                # Dictionary for daily request limits
                    "current": self.req_day_counter,                                                                    # Current request counter for the day
                    "limit": format_limit(self.req_per_day),                                                            # Maximum request limit for the day
                    "display": f"{self.req_day_counter}/{format_limit(self.req_per_day)}",                              # Display string for the current and maximum request limits
                },
            },
            "tokens": {                                                                                                 # Dictionary for token limits
                "minute": {                                                                                             # Dictionary for minute token limits
                    "current": self.tokens_minute_counter,                                                              # Current token counter for the minute
                    "limit": format_limit(self.tokens_per_min),                                                         # Maximum token limit for the minute
                    "display": f"{self.tokens_minute_counter}/{format_limit(self.tokens_per_min)}",                     # Display string for the current and maximum token limits
                },
                "day": {                                                                                                # Dictionary for daily token limits
                    "current": self.tokens_day_counter,                                                                 # Current token counter for the day
                    "limit": format_limit(self.tokens_per_day),                                                         # Maximum token limit for the day
                    "display": f"{self.tokens_day_counter}/{format_limit(self.tokens_per_day)}",                        # Display string for the current and maximum token limits
                },
            }
        }