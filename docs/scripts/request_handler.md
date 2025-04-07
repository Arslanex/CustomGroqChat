# RequestHandler Module

The `request_handler.py` module contains the `RequestHandler` class, which is responsible for preparing and processing requests to the GROQ API. It sits between the [`GroqClient`](groq_client.md) and the [`QueueManager`](queue_manager.md), validating requests and managing token counts.

## Overview

The `RequestHandler` handles several key responsibilities:

- Validating request parameters
- Managing token counting for both prompt and completion using the [`TokenCounter`](token_counter.md)
- Checking token limits against model context windows
- Preparing request payloads for the API
- Queueing requests with the appropriate priority through the [`QueueManager`](queue_manager.md)
- Providing request cancellation capabilities

For exception handling details, see the [Exceptions Guide](../usage%20guide/exceptions.md).

## Class: RequestHandler

```python
class RequestHandler:
    def __init__(self, 
                 queue_manager: QueueManager, 
                 models_config: Dict[str, Dict[str, Any]]) -> None:
        ...
```

The `RequestHandler` class prepares and processes GROQ API requests, ensuring they meet all requirements before being queued.

### Constructor

```python
def __init__(self, 
             queue_manager: QueueManager, 
             models_config: Dict[str, Dict[str, Any]]) -> None:
```

Initializes the RequestHandler with a queue manager and model configurations.

**Parameters:**
- `queue_manager` (QueueManager): The queue manager to use for request processing
- `models_config` (Dict[str, Dict[str, Any]]): Dictionary of model configurations

### Methods

#### prepare_chat_request

```python
async def prepare_chat_request(self, 
                             model_name: str,
                             messages: List[Dict[str, str]],
                             temperature: float = 0.7,
                             max_tokens: Optional[int] = None,
                             priority: str = "low",
                             callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None) -> str:
```

Prepares and queues a chat completion request.

**Parameters:**
- `model_name` (str): Name of the model to use
- `messages` (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'
- `temperature` (float): Sampling temperature (0.0 to 1.0)
- `max_tokens` (Optional[int]): Maximum number of tokens to generate
- `priority` (str): Priority level for the request ("high", "normal", or "low")
- `callback` (Optional[Callable]): Optional async callback function to be called with the API response

**Returns:**
- `str`: Request ID that can be used to track or cancel the request

**Raises:**
- `ModelNotFoundException`: If the model is not found
- `TokenLimitExceededException`: If the token limit is exceeded

#### prepare_completion_request

```python
async def prepare_completion_request(self, 
                                  model_name: str,
                                  prompt: str,
                                  temperature: float = 0.7,
                                  max_tokens: Optional[int] = None,
                                  priority: str = "low",
                                  callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None) -> str:
```

Prepares and queues a text completion request.

**Parameters:**
- `model_name` (str): Name of the model to use
- `prompt` (str): Text prompt
- `temperature` (float): Sampling temperature (0.0 to 1.0)
- `max_tokens` (Optional[int]): Maximum number of tokens to generate
- `priority` (str): Priority level for the request ("high", "normal", or "low")
- `callback` (Optional[Callable]): Optional async callback function to be called with the API response

**Returns:**
- `str`: Request ID that can be used to track or cancel the request

**Raises:**
- `ModelNotFoundException`: If the model is not found
- `TokenLimitExceededException`: If the token limit is exceeded

#### cancel_request

```python
async def cancel_request(self, request_id: str) -> bool:
```

Cancels a request by ID.

**Parameters:**
- `request_id` (str): The ID of the request to cancel

**Returns:**
- `bool`: True if the request was cancelled, False otherwise

#### get_available_models

```python
def get_available_models(self) -> List[str]:
```

Gets a list of available models.

**Returns:**
- `List[str]`: List of available model names

### Private Methods

#### _get_model_config

```python
def _get_model_config(self, model_name: str) -> Dict[str, Any]:
```

Gets the configuration for a specific model.

**Parameters:**
- `model_name` (str): Name of the model

**Returns:**
- `Dict[str, Any]`: Model configuration

**Raises:**
- `ModelNotFoundException`: If the model is not found in the configuration

#### _validate_token_limits

```python
def _validate_token_limits(self, 
                         model_name: str, 
                         token_counts: Dict[str, int],
                         max_tokens: Optional[int] = None) -> None:
```

Validates that the request is within token limits for the model.

**Parameters:**
- `model_name` (str): Name of the model
- `token_counts` (Dict[str, int]): Token count dictionary with prompt_tokens, completion_tokens, and total_tokens
- `max_tokens` (Optional[int]): Maximum number of tokens in completion (overrides max_tokens in request)

**Raises:**
- `TokenLimitExceededException`: If the token limit is exceeded

## Internal Implementation

The `RequestHandler` uses several key components:

1. **[TokenCounter](token_counter.md)**: For estimating token usage of requests
2. **[QueueManager](queue_manager.md)**: For queuing requests with appropriate priorities
3. **Model Configuration**: For validating requests against model limits

## Usage Example

```python
# This is typically used internally by the GroqClient class
async def example_usage(queue_manager, models_config):
    # Initialize handler
    request_handler = RequestHandler(queue_manager, models_config)
    
    # Prepare a chat request
    request_id = await request_handler.prepare_chat_request(
        model_name="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the capital of France?"}
        ],
        priority="high",
        callback=async_callback_function
    )
    
    # Later, if needed, cancel the request
    success = await request_handler.cancel_request(request_id)
```

## Relationship to Other Components

The `RequestHandler` sits between the [`GroqClient`](groq_client.md) and the [`QueueManager`](queue_manager.md):

- It receives requests from the [`GroqClient`](groq_client.md)
- It validates requests using model configurations
- It checks token limits using the [`TokenCounter`](token_counter.md)
- It forwards valid requests to the [`QueueManager`](queue_manager.md)
- It provides an interface for cancelling requests

## Related Documentation

- [GroqClient Documentation](groq_client.md) - Main client interface
- [QueueManager Documentation](queue_manager.md) - Request queue management
- [TokenCounter Documentation](token_counter.md) - Token counting utilities
- [Implementation Examples](../usage%20guide/implementation_examples.md) - Usage examples
- [Package Exports](package_exports.md) - Complete list of package exports 