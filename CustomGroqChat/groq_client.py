"""
GroqClient class for interacting with the GROQ Cloud API.

This is the main interface for the CustomGroqChat package, providing methods for
chat completions, text completions, and managing API requests.
"""
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
import os
import json

from .api_client import APIClient
from .config_loader import ConfigLoader
from .rate_limit_handler import RateLimitHandler
from .queue_manager import QueueManager
from .request_handler import RequestHandler
from .exceptions import CustomGroqChatException


class GroqClient:
    """
    Main client for interacting with the GROQ Cloud API.
    
    This class provides a high-level interface for making requests to the GROQ API,
    including chat completions and text completions. It handles configuration loading,
    rate limiting, token counting, and request queueing.
    """
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the GroqClient with an optional configuration file path.
        
        Args:
            config_path: Path to the configuration file. If not provided, the client
                will look for a config.json file in the current directory, or use
                environment variables.
        """
        self.config_path = config_path or os.getenv("GROQ_CONFIG_PATH") or os.path.join(os.getcwd(), "config.json")
        self.api_client = None
        self.queue_manager = None
        self.request_handler = None
        self.running = False
        self.models_config = {}
    
    async def initialize(self) -> None:
        """
        Initialize the client by loading configuration and setting up components.
        
        This method must be called before making any API requests.
        
        Raises:
            CustomGroqChatException: If initialization fails
        """
        try:
            # Load configuration
            config_loader = ConfigLoader(self.config_path)
            self.models_config = config_loader.load_config()
            
            # Get the first model's configuration for API client setup
            # (base_url and api_key are the same for all models)
            if not self.models_config:
                raise CustomGroqChatException("No models found in configuration")
            
            first_model = next(iter(self.models_config.values()))
            
            # Initialize API client
            self.api_client = APIClient(
                base_url=first_model.get("base_url"),
                api_key=first_model.get("api_key")
            )
            
            # Initialize components for each model
            for model_name, model_config in self.models_config.items():
                # Create a rate limit handler for this model
                rate_limit_handler = RateLimitHandler(model_config)
                
                # Create a queue manager for this model
                self.queue_manager = QueueManager(
                    api_client=self.api_client,
                    rate_limit_handler=rate_limit_handler
                )
                
                # Start the queue manager
                self.queue_manager.start()
            
            # Create a request handler using the queue manager
            self.request_handler = RequestHandler(
                queue_manager=self.queue_manager,
                models_config=self.models_config
            )
            
            self.running = True
            
        except Exception as e:
            # Convert to a CustomGroqChatException if it's not already one
            if not isinstance(e, CustomGroqChatException):
                raise CustomGroqChatException(f"Failed to initialize GroqClient: {str(e)}")
            raise
    
    async def chat_completion(self,
                           model_name: str,
                           messages: List[Dict[str, str]],
                           temperature: float = 0.7,
                           max_tokens: Optional[int] = None,
                           priority: str = "low",
                           stream: bool = False) -> Dict[str, Any]:
        """
        Generate a chat completion.
        
        Args:
            model_name: Name of the model to use
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            priority: Priority level for the request ("high", "normal", or "low")
            stream: Whether to stream the response (not yet implemented)
            
        Returns:
            Dict[str, Any]: The API response
            
        Raises:
            CustomGroqChatException: If an error occurs during the API call
        """
        if not self.running:
            await self.initialize()
        
        # Create a future to receive the response
        response_future = asyncio.Future()
        
        # Create a callback to set the future's result
        async def callback(response: Dict[str, Any]) -> None:
            response_future.set_result(response)
        
        # Queue the request
        await self.request_handler.prepare_chat_request(
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            priority=priority,
            callback=callback
        )
        
        # Wait for the response
        response = await response_future
        
        # Check for errors in the response
        if "error" in response:
            raise CustomGroqChatException(f"API call failed: {response['error']}")
        
        return response
    
    async def text_completion(self,
                           model_name: str,
                           prompt: str,
                           temperature: float = 0.7,
                           max_tokens: Optional[int] = None,
                           priority: str = "low",
                           stream: bool = False) -> Dict[str, Any]:
        """
        Generate a text completion.
        
        Args:
            model_name: Name of the model to use
            prompt: Text prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            priority: Priority level for the request ("high", "normal", or "low")
            stream: Whether to stream the response (not yet implemented)
            
        Returns:
            Dict[str, Any]: The API response
            
        Raises:
            CustomGroqChatException: If an error occurs during the API call
        """
        if not self.running:
            await self.initialize()
        
        # Create a future to receive the response
        response_future = asyncio.Future()
        
        # Create a callback to set the future's result
        async def callback(response: Dict[str, Any]) -> None:
            response_future.set_result(response)
        
        # Queue the request
        await self.request_handler.prepare_completion_request(
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            priority=priority,
            callback=callback
        )
        
        # Wait for the response
        response = await response_future
        
        # Check for errors in the response
        if "error" in response:
            raise CustomGroqChatException(f"API call failed: {response['error']}")
        
        return response
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get the current status of the request queue.
        
        Returns:
            Dict[str, Any]: Queue status information
        """
        if not self.running:
            await self.initialize()
        
        return self.queue_manager.get_queue_status()
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models from the configuration.
        
        Returns:
            List[str]: List of available model names
        """
        if not self.running:
            # Load configuration but don't initialize everything
            config_loader = ConfigLoader(self.config_path)
            self.models_config = config_loader.load_config()
        
        return list(self.models_config.keys())
    
    async def close(self) -> None:
        """
        Close the client and clean up resources.
        
        This method should be called when the client is no longer needed.
        """
        if self.running:
            if self.queue_manager:
                self.queue_manager.stop()
            
            if self.api_client:
                await self.api_client.close()
            
            self.running = False
