"""
API Client module for making HTTP requests to the Groq API.

This module provides a simple client for interacting with the Groq Cloud API,
handling authentication, connection management, and error handling.
"""
import json
import aiohttp
from typing import Any, Dict, Optional, Union

from .exceptions import APICallException


class APIClient:
    """
    A simple API client for making HTTP requests to the Groq API.
    """
    def __init__(self, base_url: str, api_key: str) -> None:
        """
        Initialize the API client with the base URL and API key.

        Args:
            base_url (str): The base URL for the API.
            api_key (str): The API key for authentication.
        """
        self.base_url = base_url                                                                                        # Set the base URL for the API
        self.api_key = api_key                                                                                          # Set the API key for authentication
        self.session: Optional[aiohttp.ClientSession] = None                                                            # Initialize the session to None

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get an aiohttp session. If one does not exist, create it.

        Returns:
            aiohttp.ClientSession: The aiohttp session.
        """
        if self.session is None or self.session.closed:                                                                 # Check if the session is None or closed
            self.session = aiohttp.ClientSession(                                                                       # Create a new aiohttp session
                headers={                                                                                               # Set the headers for the session
                    "Authorization": f"Bearer {self.api_key}",                                                          # Set the Authorization header with the API key
                    "Content-Type": "application/json"                                                                  # Set the Content-Type header to application/json
                }
            )

        return self.session                                                                                             # Return the session

    async def post_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a POST request to the API.

        Args:
            endpoint (str): The API endpoint to make the request to.
            payload (Dict[str, Any]): The payload to send in the request.

        Returns:
            Dict[str, Any]: The response from the API.

        Raises:
            APICallException: If there is an error making the API call.
        """
        session = await self._get_session()                                                                             # Get the aiohttp session
        url = f"{self.base_url}/{endpoint}"                                                                             # Construct the URL for the API endpoint

        try:                                                                                                            # Try to make the request
            async with session.post(url, json=payload) as response:                                                     # Make the POST request
                response.text = await response.text()                                                                   # Get the response text

                try:
                    response_data = json.loads(response.text)                                                           # Parse the response text as JSON
                except json.decoder.JSONDecodeError:                                                                    # If the response is not valid JSON, raise an exception
                    raise APICallException(                                                                             # Raise an exception if the response is not valid JSON
                        message=f"Failed to decode JSON response: {response.text}",                                      # Set the error message
                        status_code=response.status                                                                     # Set the status code
                        )

                # Check for API errors
                if response.status >= 400:                                                                              # If the response status is 400 or higher
                    raise APICallException(                                                                             # Raise an exception
                        message=f"API call failed with status {response.status}: {response_data}",                      # Set the error message
                        status_code=response.status                                                                     # Set the status code
                    )
                
                return response_data                                                                                    # Return the response data

        except aiohttp.ClientError as e:                                                                                # If there is a client error
            raise APICallException(                                                                                     # Raise an exception if there is a client error
                message=f"Client error: {str(e)}",                                                                      # Set the error message
                status_code=500                                                                                         # Set the status code to 500
            )

    async def close(self) -> None:
        """
        Close the aiohttp session if it exists
        """
        if self.session and not self.session.closed:                                                                    # Check if the session exists and is not closed
            await self.session.close()                                                                                  # Close the session