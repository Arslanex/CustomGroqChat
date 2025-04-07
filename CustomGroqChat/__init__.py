"""
CustomGroqChat - A third-party client for the Groq Chat API.

This module provides a straightforward interface for interacting with the Groq Chat API,
handling rate limiting, token counting, and request management.

Basic usage:
    
    # Initialize
    from CustomGroqChat import GroqClient
    import asyncio
    
    async def main():
        # Initialize with config
        client = GroqClient("config.json")
        await client.initialize()
        
        # Chat completion
        response = await client.chat_completion(
            model_name="llama3-8b-preview",
            messages=[{"role": "user", "content": "Tell me a joke"}]
        )
        print(response["choices"][0]["message"]["content"])
        
        # Close the client
        await client.close()
    
    asyncio.run(main())
"""

# Import the main client class
from .groq_client import GroqClient

# Import core components for advanced usage
from .request_handler import RequestHandler
from .api_client import APIClient
from .config_loader import ConfigLoader
from .rate_limit_handler import RateLimitHandler
from .queue_manager import QueueManager

# Import token counting utilities
from .token_counter import (
    count_tokens_in_message,
    count_tokens_in_messages,
    count_tokens_in_prompt,
    count_tokens_in_request,
    count_request_and_completion_tokens
)

# Import exceptions
from .exceptions import (
    CustomGroqChatException,
    ConfigLoaderException,
    RateLimitExceededException,
    APICallException,
    ModelNotFoundException,
    TokenLimitExceededException
)

# Set up __all__ to control what's imported with wildcard imports
__all__ = [
    # Main client class
    'GroqClient',
    
    # Core classes for advanced usage
    'RequestHandler',
    'APIClient',
    'ConfigLoader',
    'RateLimitHandler',
    'QueueManager',
    
    # Token counting functions
    'count_tokens_in_message',
    'count_tokens_in_messages',
    'count_tokens_in_prompt', 
    'count_tokens_in_request',
    'count_request_and_completion_tokens',
    
    # Exceptions
    'CustomGroqChatException',
    'ConfigLoaderException',
    'RateLimitExceededException',
    'APICallException',
    'ModelNotFoundException',
    'TokenLimitExceededException',
]
