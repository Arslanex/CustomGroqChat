"""
RequestHandler class for preparing and processing GROQ API requests.

This module handles the preparation of API requests, validation of parameters,
and processing of API responses.
"""
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable

from .token_counter import count_request_and_completion_tokens
from .queue_manager import QueueManager
from .exceptions import TokenLimitExceededException, ModelNotFoundException


class RequestHandler:
    """
    Handles preparation and processing of GROQ API requests.
    
    This class is responsible for validating request parameters,
    managing token counts, and queueing requests.
    """
    def __init__(self, 
                 queue_manager: QueueManager, 
                 models_config: Dict[str, Dict[str, Any]]) -> None:
        """
        Initialize the RequestHandler with a queue manager and model configurations.
        
        Args:
            queue_manager: The queue manager to use for request processing
            models_config: Dictionary of model configurations
        """
        self.queue_manager = queue_manager
        self.models_config = models_config

    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get the configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dict[str, Any]: Model configuration
            
        Raises:
            ModelNotFoundException: If the model is not found in the configuration
        """
        model_config = self.models_config.get(model_name)
        if not model_config:
            raise ModelNotFoundException(
                message=f"Model '{model_name}' not found in configuration",
                model_name=model_name
            )
        return model_config

    def _validate_token_limits(self, 
                             model_name: str, 
                             token_counts: Dict[str, int],
                             max_tokens: Optional[int] = None) -> None:
        """
        Validate that the request is within token limits for the model.
        
        Args:
            model_name: Name of the model
            token_counts: Token count dictionary with prompt_tokens, completion_tokens, and total_tokens
            max_tokens: Maximum number of tokens in completion (overrides max_tokens in request)
            
        Raises:
            TokenLimitExceededException: If the token limit is exceeded
        """
        model_config = self._get_model_config(model_name)
        
        # Check if the model has a maximum context size
        context_window = model_config.get("context_window", 0)
        if context_window > 0 and token_counts["total_tokens"] > context_window:
            raise TokenLimitExceededException(
                message=f"Token limit exceeded for model {model_name}. " 
                        f"Total tokens: {token_counts['total_tokens']}, "
                        f"Context window: {context_window}",
                token_count=token_counts["total_tokens"],
                token_limit=context_window,
                model_name=model_name
            )

    async def prepare_chat_request(self, 
                                 model_name: str,
                                 messages: List[Dict[str, str]],
                                 temperature: float = 0.7,
                                 max_tokens: Optional[int] = None,
                                 priority: str = "low",
                                 callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None) -> str:
        """
        Prepare and queue a chat completion request.
        
        Args:
            model_name: Name of the model to use
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            priority: Priority level for the request ("high", "normal", or "low")
            callback: Optional async callback function to be called with the API response
            
        Returns:
            str: Request ID that can be used to track or cancel the request
            
        Raises:
            ModelNotFoundException: If the model is not found
            TokenLimitExceededException: If the token limit is exceeded
        """
        # Validate that the model exists
        model_config = self._get_model_config(model_name)
        
        # Prepare the request payload
        request_data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature
        }
        
        # Add max_tokens if provided
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        
        # Count tokens
        token_counts = count_request_and_completion_tokens(request_data, model_name)
        
        # Validate token limits
        self._validate_token_limits(model_name, token_counts, max_tokens)
        
        # Queue the request
        request_id = await self.queue_manager.enqueue_request(
            endpoint="chat/completions",
            payload=request_data,
            token_count=token_counts["total_tokens"],
            callback=callback,
            priority=priority
        )
        
        return request_id

    async def prepare_completion_request(self, 
                                      model_name: str,
                                      prompt: str,
                                      temperature: float = 0.7,
                                      max_tokens: Optional[int] = None,
                                      priority: str = "low",
                                      callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None) -> str:
        """
        Prepare and queue a text completion request.
        
        Args:
            model_name: Name of the model to use
            prompt: Text prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            priority: Priority level for the request ("high", "normal", or "low")
            callback: Optional async callback function to be called with the API response
            
        Returns:
            str: Request ID that can be used to track or cancel the request
            
        Raises:
            ModelNotFoundException: If the model is not found
            TokenLimitExceededException: If the token limit is exceeded
        """
        # Validate that the model exists
        model_config = self._get_model_config(model_name)
        
        # Prepare the request payload
        request_data = {
            "model": model_name,
            "prompt": prompt,
            "temperature": temperature
        }
        
        # Add max_tokens if provided
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        
        # Count tokens
        token_counts = count_request_and_completion_tokens(request_data, model_name)
        
        # Validate token limits
        self._validate_token_limits(model_name, token_counts, max_tokens)
        
        # Queue the request
        request_id = await self.queue_manager.enqueue_request(
            endpoint="completions",
            payload=request_data,
            token_count=token_counts["total_tokens"],
            callback=callback,
            priority=priority
        )
        
        return request_id

    async def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a request by ID.
        
        Args:
            request_id: The ID of the request to cancel
            
        Returns:
            bool: True if the request was cancelled, False otherwise
        """
        return await self.queue_manager.cancel_request(request_id)

    def get_available_models(self) -> List[str]:
        """
        Get a list of available models.
        
        Returns:
            List[str]: List of available model names
        """
        return list(self.models_config.keys())
