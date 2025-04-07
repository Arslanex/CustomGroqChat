# Error Handling Guide

This document explains the exception handling in CustomGroqChat and how to handle different types of errors.

## Exception Hierarchy

CustomGroqChat provides a hierarchy of exceptions to help you identify and handle different error conditions:

```
CustomGroqChatException (base exception)
├── ConfigurationError
├── InitializationError
├── APIError
│   ├── RateLimitError
│   ├── AuthenticationError
│   └── ServiceUnavailableError
├── RequestError
├── TokenBudgetExceededError
└── ModelNotFoundError
```

## Core Exceptions

### CustomGroqChatException

The base exception that all other exceptions inherit from. You can catch this to handle any error from the library.

```python
from CustomGroqChat import GroqClient, CustomGroqChatException

async def handle_all_errors():
    client = GroqClient()
    
    try:
        await client.initialize()
        response = await client.chat_completion(
            model_name="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    except CustomGroqChatException as e:
        print(f"An error occurred: {e}")
    finally:
        await client.close()
```

### ConfigurationError

Raised when there's an issue with the configuration file or configuration parameters.

Common causes:
- Missing or invalid configuration file
- Malformed JSON in configuration
- Missing required fields in model configuration

```python
from CustomGroqChat import GroqClient, ConfigurationError

async def handle_config_error():
    try:
        client = GroqClient("invalid_path.json")
        await client.initialize()
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        # You could create a default configuration here
```

### InitializationError

Raised when the client fails to initialize properly.

Common causes:
- Network issues during initialization
- Invalid API credentials
- Invalid model configuration

```python
from CustomGroqChat import GroqClient, InitializationError

async def handle_init_error():
    client = GroqClient()
    try:
        await client.initialize()
    except InitializationError as e:
        print(f"Failed to initialize client: {e}")
        # Prompt user to check their API key or configuration
```

## API Errors

### APIError

Base class for all errors related to API communication.

### RateLimitError

Raised when rate limits are exceeded and the request queue can't handle it. This is rare since the client handles most rate limiting internally, but can occur in extreme cases.

```python
from CustomGroqChat import GroqClient, RateLimitError

async def handle_rate_limit():
    client = GroqClient()
    await client.initialize()
    
    try:
        # Attempting too many requests at once
        tasks = []
        for i in range(100):  # Very aggressive request pattern
            messages = [{"role": "user", "content": f"Question {i}"}]
            task = client.chat_completion(
                model_name="llama-3.1-8b-instant",
                messages=messages
            )
            tasks.append(task)
        
        # Wait for all to complete
        await asyncio.gather(*tasks)
        
    except RateLimitError as e:
        print(f"Rate limit exceeded: {e}")
        print(f"Please wait before sending more requests.")
        
        # Check current limits
        limits = await client.check_model_limits("llama-3.1-8b-instant")
        print(f"Current usage: {limits['minute_used']}/{limits['minute_limit']} requests this minute")
        
        # Calculate wait time
        wait_time = 60  # seconds (until the next minute window)
        print(f"Waiting for {wait_time} seconds before retrying...")
        await asyncio.sleep(wait_time)
        
        # Retry with fewer requests
        # ...
    finally:
        await client.close()
```

### AuthenticationError

Raised when there are issues with authentication to the API.

Common causes:
- Invalid API key
- Expired API key
- API key lacks necessary permissions

```python
from CustomGroqChat import GroqClient, AuthenticationError

async def handle_auth_error():
    client = GroqClient()
    
    try:
        await client.initialize()
        response = await client.chat_completion(
            model_name="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    except AuthenticationError as e:
        print(f"Authentication failed: {e}")
        # Prompt user to update their API key
    finally:
        await client.close()
```

### ServiceUnavailableError

Raised when the API service is temporarily unavailable.

```python
from CustomGroqChat import GroqClient, ServiceUnavailableError
import asyncio

async def handle_service_unavailable():
    client = GroqClient()
    await client.initialize()
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = await client.chat_completion(
                model_name="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Hello!"}]
            )
            print(response["choices"][0]["message"]["content"])
            break
            
        except ServiceUnavailableError as e:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                print(f"Service unavailable: {e}. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                print(f"Service still unavailable after {max_retries} attempts.")
    
    await client.close()
```

## Other Exceptions

### RequestError

General error related to request formatting or processing.

Common causes:
- Invalid message format
- Invalid parameters
- Request timed out

```python
from CustomGroqChat import GroqClient, RequestError

async def handle_request_error():
    client = GroqClient()
    await client.initialize()
    
    try:
        # Incorrectly formatted message
        response = await client.chat_completion(
            model_name="llama-3.1-8b-instant",
            messages={"role": "user", "content": "Hello!"}  # Should be a list!
        )
    except RequestError as e:
        print(f"Request error: {e}")
        # Fix the request format and retry
    finally:
        await client.close()
```

### TokenBudgetExceededError

Raised when a request would exceed the token budget/context window of the model.

```python
from CustomGroqChat import GroqClient, TokenBudgetExceededError

async def handle_token_budget_error():
    client = GroqClient()
    await client.initialize()
    
    try:
        # Create a very long message
        long_content = "This is a test. " * 2000  # Very long content
        
        response = await client.chat_completion(
            model_name="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": long_content}]
        )
    except TokenBudgetExceededError as e:
        print(f"Token budget exceeded: {e}")
        print(f"Maximum context window: {e.max_tokens}")
        print(f"Estimated tokens in request: {e.estimated_tokens}")
        
        # Truncate the message to fit within limits
        # ...
    finally:
        await client.close()
```

### ModelNotFoundError

Raised when attempting to use a model that doesn't exist in the configuration.

```python
from CustomGroqChat import GroqClient, ModelNotFoundError

async def handle_model_not_found():
    client = GroqClient()
    await client.initialize()
    
    try:
        response = await client.chat_completion(
            model_name="nonexistent-model",  # This model doesn't exist in config
            messages=[{"role": "user", "content": "Hello!"}]
        )
    except ModelNotFoundError as e:
        print(f"Model not found: {e}")
        
        # Show available models
        available_models = client.get_available_models()
        print(f"Available models: {available_models}")
        
        # Use a valid model instead
        if available_models:
            default_model = available_models[0]
            print(f"Using default model: {default_model}")
            response = await client.chat_completion(
                model_name=default_model,
                messages=[{"role": "user", "content": "Hello!"}]
            )
    finally:
        await client.close()
```

## Combining Exception Handling

In a real application, you'll likely want to handle multiple exception types:

```python
import asyncio
from CustomGroqChat import (
    GroqClient,
    CustomGroqChatException,
    ConfigurationError, 
    InitializationError,
    APIError,
    RateLimitError,
    AuthenticationError,
    ServiceUnavailableError,
    RequestError,
    TokenBudgetExceededError,
    ModelNotFoundError
)

async def comprehensive_error_handling():
    try:
        # Initialize client
        client = GroqClient("config.json")
        try:
            await client.initialize()
        except (ConfigurationError, InitializationError) as e:
            print(f"Setup error: {e}")
            return
        
        # Prepare request
        try:
            # Check if model exists
            model_name = "llama-3.1-8b-instant"
            available_models = client.get_available_models()
            if model_name not in available_models:
                raise ModelNotFoundError(f"Model {model_name} not found in configuration")
            
            # Prepare messages
            messages = [{"role": "user", "content": "Hello!"}]
            
            # Check token count (optional)
            try:
                token_count = await client.count_tokens(model_name, messages)
                print(f"Estimated token count: {token_count}")
            except Exception as e:
                print(f"Token counting failed, proceeding anyway: {e}")
            
            # Send request
            try:
                response = await client.chat_completion(
                    model_name=model_name,
                    messages=messages
                )
                print(response["choices"][0]["message"]["content"])
            except RateLimitError as e:
                print(f"Rate limit exceeded: {e}. Waiting and retrying...")
                await asyncio.sleep(5)
                # Retry logic here
            except AuthenticationError as e:
                print(f"Authentication failed: {e}")
                # Prompt for new API key
            except ServiceUnavailableError as e:
                print(f"Service unavailable: {e}")
                # Implement retry with backoff
            except TokenBudgetExceededError as e:
                print(f"Token budget exceeded: {e}")
                # Truncate messages or use a different approach
            except RequestError as e:
                print(f"Request error: {e}")
                # Fix request and retry
            except APIError as e:
                print(f"API error: {e}")
                # General API error handling
                
        except ModelNotFoundError as e:
            print(f"Model error: {e}")
            print(f"Available models: {available_models}")
            
    except CustomGroqChatException as e:
        print(f"Unexpected error: {e}")
    finally:
        # Always close the client, if it was initialized
        if 'client' in locals():
            await client.close()

# Run the example
asyncio.run(comprehensive_error_handling())
```

## Best Practices for Error Handling

1. **Use specific exception types** when you need to handle specific errors differently.

2. **Catch the base exception** (`CustomGroqChatException`) as a fallback for unexpected errors.

3. **Always close the client** in a `finally` block to ensure resources are cleaned up.

4. **Implement retry logic with exponential backoff** for transient errors.

5. **Log errors** to help with debugging and monitoring.

6. **Provide helpful error messages** to users when something goes wrong.

7. **Consider graceful degradation** - if one model fails, try another or offer a simplified experience.

8. **Check token counts** before sending requests to avoid TokenBudgetExceededError. 