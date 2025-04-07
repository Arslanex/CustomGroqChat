# CustomGroqChat Package Exports

This document details the public exports provided by the CustomGroqChat package through its `__init__.py` file. Understanding these exports is crucial for effectively using the package in your applications.

## Overview

The CustomGroqChat package is designed with a layered API:

1. **Simple API**: Most users will only need to import the `GroqClient` class
2. **Advanced API**: For more complex use cases, individual components are also exported
3. **Utility Functions**: Token counting utilities are exposed for advanced token management
4. **Exceptions**: Custom exceptions are exported for detailed error handling

## Exported Classes and Functions

### Main Client Class

| Export | Description | Documentation |
|--------|-------------|---------------|
| `GroqClient` | Main interface for interacting with GROQ API | [GroqClient Documentation](groq_client.md) |

### Core Components (for Advanced Usage)

| Export | Description | Documentation |
|--------|-------------|---------------|
| `RequestHandler` | Prepares and validates API requests | [RequestHandler Documentation](request_handler.md) |
| `APIClient` | Handles direct communication with GROQ API | [APIClient Documentation](api_client.md) |
| `ConfigLoader` | Loads and validates configuration | [ConfigLoader Documentation](config_loader.md) |
| `RateLimitHandler` | Manages API rate limits | [RateLimitHandler Documentation](rate_limit_handler.md) |
| `QueueManager` | Manages request queues and prioritization | [QueueManager Documentation](queue_manager.md) |

### Token Counting Utilities

| Export | Description |
|--------|-------------|
| `count_tokens_in_message` | Counts tokens in a single message |
| `count_tokens_in_messages` | Counts tokens in a list of messages |
| `count_tokens_in_prompt` | Counts tokens in a text prompt |
| `count_tokens_in_request` | Counts tokens in a full request |
| `count_request_and_completion_tokens` | Counts tokens in both request and expected completion |

See [TokenCounter Documentation](token_counter.md) for details on these functions.

### Exceptions

| Export | Description |
|--------|-------------|
| `CustomGroqChatException` | Base exception for all package exceptions |
| `ConfigLoaderException` | Raised when configuration loading fails |
| `RateLimitExceededException` | Raised when rate limits are exceeded |
| `APICallException` | Raised when API calls fail |
| `ModelNotFoundException` | Raised when a requested model is not found |
| `TokenLimitExceededException` | Raised when token limits are exceeded |

## Usage Examples

### Basic Usage (Most Common)

```python
from CustomGroqChat import GroqClient
import asyncio

async def main():
    # Initialize with config
    client = GroqClient("config.json")
    await client.initialize()
    
    # Chat completion
    response = await client.chat_completion(
        model_name="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Tell me a joke"}]
    )
    print(response["choices"][0]["message"]["content"])
    
    # Close the client
    await client.close()

asyncio.run(main())
```

### Advanced Usage (Custom Components)

```python
from CustomGroqChat import ConfigLoader, APIClient, RateLimitHandler, QueueManager, RequestHandler
import asyncio

async def advanced_example():
    # Load configuration
    config_loader = ConfigLoader("config.json")
    models_config = config_loader.load_config()
    
    # Get first model config
    first_model = next(iter(models_config.values()))
    
    # Set up components
    api_client = APIClient(
        base_url=first_model.get("base_url"),
        api_key=first_model.get("api_key")
    )
    
    rate_limit_handler = RateLimitHandler(first_model)
    
    queue_manager = QueueManager(
        api_client=api_client,
        rate_limit_handler=rate_limit_handler
    )
    queue_manager.start()
    
    request_handler = RequestHandler(
        queue_manager=queue_manager,
        models_config=models_config
    )
    
    # Custom request handling
    try:
        # Use the custom components
        # ...
    finally:
        queue_manager.stop()
        await api_client.close()
```

### Token Counting

```python
from CustomGroqChat import count_tokens_in_messages, count_request_and_completion_tokens

# Count tokens in messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
]
token_count = count_tokens_in_messages(messages, "llama-3.1-8b-instant")
print(f"Message tokens: {token_count}")

# Count tokens in a complete request
request_data = {
    "model": "llama-3.1-8b-instant",
    "messages": messages,
    "temperature": 0.7
}
token_counts = count_request_and_completion_tokens(request_data, "llama-3.1-8b-instant")
print(f"Request tokens: {token_counts['prompt_tokens']}")
print(f"Expected completion tokens: {token_counts['completion_tokens']}")
print(f"Total tokens: {token_counts['total_tokens']}")
```

### Error Handling

```python
from CustomGroqChat import GroqClient, CustomGroqChatException, ModelNotFoundException, TokenLimitExceededException
import asyncio

async def error_handling_example():
    client = GroqClient("config.json")
    await client.initialize()
    
    try:
        # Try to use a non-existent model
        response = await client.chat_completion(
            model_name="nonexistent-model",
            messages=[{"role": "user", "content": "Hello"}]
        )
    except ModelNotFoundException as e:
        print(f"Model not found: {e.model_name}")
    except TokenLimitExceededException as e:
        print(f"Token limit exceeded: {e.token_count}/{e.token_limit}")
    except CustomGroqChatException as e:
        print(f"General error: {e}")
    finally:
        await client.close()

asyncio.run(error_handling_example())
```

## Related Documentation

- [GroqClient Documentation](groq_client.md) - Main client interface
- [TokenCounter Documentation](token_counter.md) - Token counting utilities
- [Examples](../examples.md) - Complete example scripts 