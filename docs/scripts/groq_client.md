# GroqClient Module

The `groq_client.py` module contains the main client interface for the CustomGroqChat package. It provides the primary entry point for applications interacting with the GROQ API.

## Overview

The `GroqClient` class serves as the central component of the CustomGroqChat package, orchestrating all the other components to provide a seamless interface for making requests to the GROQ API. It handles:

- Configuration loading using [ConfigLoader](config_loader.md)
- Initialization of all components
- Rate limiting management through [RateLimitHandler](rate_limit_handler.md)
- Token counting and validation via [TokenCounter](token_counter.md)
- Request queuing and prioritization with [QueueManager](queue_manager.md)
- Request preparation via [RequestHandler](request_handler.md)
- API communication through [APIClient](api_client.md)
- Error handling using custom [exceptions](../usage%20guide/exceptions.md)

For a complete overview of the package's public API, see [Package Exports](package_exports.md).

## Class: GroqClient

```python
class GroqClient:
    def __init__(self, config_path: Optional[str] = None) -> None:
        ...
```

The main client for interacting with the GROQ API. It orchestrates all components and provides a high-level interface for making API requests.

### Constructor

```python
def __init__(self, config_path: Optional[str] = None) -> None:
```

Initializes the GroqClient with an optional configuration file path.

**Parameters:**
- `config_path` (Optional[str]): Path to the configuration file. If not provided, the client will look for a `config.json` file in the current directory, or use environment variables.

### Methods

#### initialize

```python
async def initialize(self) -> None:
```

Initializes the client by loading configuration and setting up all required components. This method must be called before making any API requests.

**Raises:**
- `CustomGroqChatException`: If initialization fails

#### chat_completion

```python
async def chat_completion(
    self,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    priority: str = "low",
    stream: bool = False
) -> Dict[str, Any]:
```

Generates a chat completion using the specified model and parameters.

**Parameters:**
- `model_name` (str): Name of the model to use
- `messages` (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'
- `temperature` (float): Sampling temperature (0.0 to 1.0)
- `max_tokens` (Optional[int]): Maximum number of tokens to generate
- `priority` (str): Priority level for the request ("high", "normal", or "low")
- `stream` (bool): Whether to stream the response (not yet fully implemented)

**Returns:**
- `Dict[str, Any]`: The API response

**Raises:**
- `CustomGroqChatException`: If an error occurs during the API call

#### text_completion

```python
async def text_completion(
    self,
    model_name: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    priority: str = "low",
    stream: bool = False
) -> Dict[str, Any]:
```

Generates a text completion using the specified model and parameters.

**Parameters:**
- `model_name` (str): Name of the model to use
- `prompt` (str): Text prompt
- `temperature` (float): Sampling temperature (0.0 to 1.0)
- `max_tokens` (Optional[int]): Maximum number of tokens to generate
- `priority` (str): Priority level for the request ("high", "normal", or "low")
- `stream` (bool): Whether to stream the response (not yet fully implemented)

**Returns:**
- `Dict[str, Any]`: The API response

**Raises:**
- `CustomGroqChatException`: If an error occurs during the API call

#### get_queue_status

```python
async def get_queue_status(self) -> Dict[str, Any]:
```

Gets the current status of the request queue.

**Returns:**
- `Dict[str, Any]`: Queue status information including pending requests, active requests, and rate limit information

#### get_available_models

```python
def get_available_models(self) -> List[str]:
```

Gets a list of available models from the configuration.

**Returns:**
- `List[str]`: List of available model names

#### close

```python
async def close(self) -> None:
```

Closes the client and cleans up resources. This method should be called when the client is no longer needed, typically in a finally block.

## Internal Implementation

The `GroqClient` class orchestrates several internal components:

1. **[ConfigLoader](config_loader.md)**: Loads and validates configuration from a file or environment variables
2. **[APIClient](api_client.md)**: Handles low-level HTTP requests to the GROQ API
3. **[RateLimitHandler](rate_limit_handler.md)**: Tracks and manages rate limits for API requests
4. **[QueueManager](queue_manager.md)**: Manages the request queue and handles request prioritization
5. **[RequestHandler](request_handler.md)**: Prepares and validates API requests

## Usage Example

```python
import asyncio
from CustomGroqChat import GroqClient

async def main():
    # Initialize client
    client = GroqClient("config.json")
    await client.initialize()
    
    try:
        # Generate a chat completion
        response = await client.chat_completion(
            model_name="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the capital of France?"}
            ]
        )
        
        # Print the response
        print(response["choices"][0]["message"]["content"])
        
    finally:
        # Clean up resources
        await client.close()

asyncio.run(main())
```

## Relationship to Other Components

The `GroqClient` is the central component that coordinates all other parts of the system:

- It uses **ConfigLoader** to load the configuration
- It initializes the **APIClient** for making HTTP requests
- It creates a **RateLimitHandler** for each model to manage rate limits
- It initializes the **QueueManager** to handle request queuing
- It creates a **RequestHandler** to prepare and validate requests
- It handles errors through the **CustomGroqChatException** hierarchy 

## Related Documentation

- [Package Exports](package_exports.md) - Complete list of package exports
- [Implementation Examples](../usage%20guide/implementation_examples.md) - Examples of using the GroqClient
- [Simple Usage Guide](../usage%20guide/simple_usage.md) - Getting started with the GroqClient
- [Advanced Usage Guide](../usage%20guide/advanced_usage.md) - Advanced usage patterns
- [Examples Overview](../examples.md) - Example scripts using this component 