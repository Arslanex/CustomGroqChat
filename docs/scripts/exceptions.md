# CustomGroqChat Exception System

## Overview

The `exceptions.py` module defines the exception hierarchy used throughout the CustomGroqChat package. These exceptions provide detailed information about errors that occur during operation, making it easier to diagnose and handle problems.

All exceptions in the package inherit from the base `CustomGroqChatException` class, which provides common functionality like error codes and dictionary conversion for logging or serialization.

## Exception Hierarchy

```
CustomGroqChatException (base exception)
├── ConfigLoaderException
├── RateLimitExceededException 
├── APICallException
├── ModelNotFoundException
└── TokenLimitExceededException
```

## Base Exception

### CustomGroqChatException

The root exception class from which all other CustomGroqChat exceptions inherit.

**Attributes:**
- `message` (str): Human-readable error message
- `error_code` (Optional[str]): Machine-readable error code

**Methods:**
- `to_dict()`: Converts the exception to a dictionary representation for logging or serialization

**Example:**
```python
try:
    # CustomGroqChat operation that might fail
    pass
except CustomGroqChatException as e:
    print(f"Error: {e.message}")
    print(f"Error code: {e.error_code}")
    error_dict = e.to_dict()  # For logging or serialization
```

## Configuration Exceptions

### ConfigLoaderException

Raised when there are issues loading or validating configuration.

**Error Code:** `CONFIG_LOADER_ERROR`

**Attributes:**
- `message` (str): Error message
- `error_code` (str): Always `CONFIG_LOADER_ERROR`
- `config_key` (Optional[str]): The configuration key that caused the error, if applicable

**Example:**
```python
from CustomGroqChat import GroqClient, ConfigLoaderException

try:
    client = GroqClient("invalid_path.json")
    await client.initialize()
except ConfigLoaderException as e:
    print(f"Configuration error: {e.message}")
    if e.config_key:
        print(f"Problem with config key: {e.config_key}")
```

## Rate Limit Exceptions

### RateLimitExceededException

Raised when API rate limits are exceeded and the request can't be queued.

**Error Code:** `RATE_LIMIT_EXCEEDED`

**Attributes:**
- `message` (str): Error message
- `error_code` (str): Always `RATE_LIMIT_EXCEEDED`
- `limit_type` (str): Type of rate limit exceeded (e.g., "tokens_per_minute")
- `current_value` (int): Current usage value
- `limit_value` (int): Maximum allowed value
- `time_period` (str): Time period for the rate limit (e.g., "minute", "day")

**Example:**
```python
from CustomGroqChat import GroqClient, RateLimitExceededException
import asyncio

async def handle_rate_limit():
    client = GroqClient()
    await client.initialize()
    
    try:
        # Send many requests in rapid succession
        tasks = []
        for i in range(100):
            task = client.chat_completion(
                model_name="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": f"Question {i}"}]
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
    except RateLimitExceededException as e:
        print(f"Rate limit exceeded: {e.message}")
        print(f"Limit type: {e.limit_type}")
        print(f"Current usage: {e.current_value}/{e.limit_value} {e.time_period}")
        
        # Wait until the rate limit resets
        wait_time = 60  # seconds (until the next minute window)
        print(f"Waiting {wait_time} seconds before retrying...")
        await asyncio.sleep(wait_time)
        
        # Retry with fewer concurrent requests
```

## API Exceptions

### APICallException

Raised when an API call fails due to communication issues, invalid requests, or server errors.

**Error Code:** `API_CALL_ERROR` or `API_CALL_ERROR_{status_code}`

**Attributes:**
- `message` (str): Error message
- `error_code` (str): Contains the status code if available
- `status_code` (Optional[int]): HTTP status code if available
- `response_body` (Optional[Dict[str, Any]]): Response body if available

**Example:**
```python
from CustomGroqChat import GroqClient, APICallException

async def handle_api_error():
    client = GroqClient()
    await client.initialize()
    
    try:
        response = await client.chat_completion(
            model_name="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hello"}]
        )
    except APICallException as e:
        print(f"API error: {e.message}")
        
        if e.status_code:
            print(f"Status code: {e.status_code}")
            
            if e.status_code >= 500:
                print("Server error - try again later")
            elif e.status_code == 401:
                print("Authentication failed - check your API key")
            elif e.status_code == 429:
                print("Too many requests - slow down")
        
        if e.response_body:
            print(f"Response details: {e.response_body}")
```

## Model Exceptions

### ModelNotFoundException

Raised when a requested model is not found or not available.

**Error Code:** `MODEL_NOT_FOUND`

**Attributes:**
- `message` (str): Error message
- `error_code` (str): Always `MODEL_NOT_FOUND`
- `model_name` (Optional[str]): Name of the model that wasn't found

**Example:**
```python
from CustomGroqChat import GroqClient, ModelNotFoundException

async def handle_model_not_found():
    client = GroqClient()
    await client.initialize()
    
    try:
        response = await client.chat_completion(
            model_name="nonexistent-model",
            messages=[{"role": "user", "content": "Hello"}]
        )
    except ModelNotFoundException as e:
        print(f"Model not found: {e.message}")
        print(f"Requested model: {e.model_name}")
        
        # Get available models and suggest an alternative
        available_models = await client.get_available_models()
        print(f"Available models: {', '.join(available_models)}")
        
        # Use an available model instead
        if available_models:
            print(f"Using {available_models[0]} instead")
            response = await client.chat_completion(
                model_name=available_models[0],
                messages=[{"role": "user", "content": "Hello"}]
            )
```

## Token Limit Exceptions

### TokenLimitExceededException

Raised when a request would exceed the token limit for a model.

**Error Code:** `TOKEN_LIMIT_EXCEEDED`

**Attributes:**
- `message` (str): Error message
- `error_code` (str): Always `TOKEN_LIMIT_EXCEEDED`
- `token_count` (int): Current token count
- `token_limit` (int): Maximum token limit
- `model_name` (str): Name of the model

**Example:**
```python
from CustomGroqChat import GroqClient, TokenLimitExceededException
from CustomGroqChat.token_counter import count_tokens_in_messages

async def handle_token_limit():
    client = GroqClient()
    await client.initialize()
    
    # Create a very long message
    long_content = "This is a test. " * 2000  # Very long content
    messages = [{"role": "user", "content": long_content}]
    
    try:
        response = await client.chat_completion(
            model_name="llama-3.1-8b-instant",
            messages=messages
        )
    except TokenLimitExceededException as e:
        print(f"Token limit exceeded: {e.message}")
        print(f"Token count: {e.token_count}/{e.token_limit}")
        print(f"Model: {e.model_name}")
        
        # Try to truncate the message to fit within limits
        max_message_tokens = e.token_limit - 100  # Leave room for the response
        
        while count_tokens_in_messages(messages, e.model_name) > max_message_tokens:
            # Truncate the content by 10%
            current_content = messages[0]["content"]
            truncated_length = int(len(current_content) * 0.9)
            messages[0]["content"] = current_content[:truncated_length]
        
        print(f"Truncated message to fit within token limit")
        
        # Retry with truncated message
        response = await client.chat_completion(
            model_name=e.model_name,
            messages=messages
        )
```

## Error Handling Best Practices

1. **Catch specific exceptions first**: Handle specific exceptions before catching the base exception to provide more targeted error handling.

2. **Log the error details**: Use the `to_dict()` method to get a complete error representation for logging.

3. **Implement retries with backoff**: For transient errors like rate limits or service unavailability, implement exponential backoff.

4. **Graceful degradation**: When a specific model is unavailable, try an alternative model or reduce the complexity of the request.

5. **Validate inputs early**: Check inputs before making API calls to catch token limit issues early.

## Related Documentation

- [Rate Limit Handler](rate_limit_handler.md) - Manages rate limits to prevent exceptions
- [Usage Guide: Error Handling](../usage%20guide/exceptions.md) - General error handling guide
- [Request Handler](request_handler.md) - Validates requests to prevent exceptions
- [GroqClient](groq_client.md) - Main client that can throw these exceptions 