# Implementation Examples

This document provides implementation examples for integrating CustomGroqChat into your applications. These examples are more focused on implementation patterns than the example scripts in the main examples directory.

> **Note:** For general information about the example scripts included with CustomGroqChat, see the [Examples Overview](../examples.md) in the main documentation.

## Basic Implementation Patterns

### 1. Simple Chat Implementation

This example demonstrates the most basic implementation of the CustomGroqChat client:

```python
import asyncio
import os
from CustomGroqChat import GroqClient

async def main():
    # Initialize the client
    client = GroqClient()
    await client.initialize()
    
    try:
        # Get list of available models
        models = client.get_available_models()
        print(f"Available models: {models}")
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the capital of France?"}
        ]
        
        # Use the first available model
        if models:
            model_name = models[0]
            
            # Generate a chat completion
            response = await client.chat_completion(
                model_name=model_name,
                messages=messages,
                temperature=0.7
            )
            
            # Print the response
            print(f"Model: {model_name}")
            print(f"Response: {response['choices'][0]['message']['content']}")
            
            # Print token usage
            print(f"Token usage: {response['usage']}")
    finally:
        # Always close the client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

**Implementation insights:**
- Basic setup and initialization of the client
- Simple message structure with system and user messages
- Automatic model selection based on available models
- Basic response handling and token usage information

### 2. Parallel Requests Implementation

This example demonstrates implementing parallel requests processing:

```python
import asyncio
import time
from CustomGroqChat import GroqClient

async def main():
    # Initialize the client
    client = GroqClient()
    await client.initialize()
    
    try:
        # Get available models
        models = client.get_available_models()
        if not models:
            print("No models available")
            return
            
        model_name = models[0]
        print(f"Using model: {model_name}")
        
        # Create 5 parallel requests
        start_time = time.time()
        tasks = []
        
        for i in range(1, 6):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Count from 1 to {i} and explain why counting is important."}
            ]
            
            task = client.chat_completion(
                model_name=model_name,
                messages=messages
            )
            tasks.append(task)
        
        # Wait for all requests to complete
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Process responses
        for i, response in enumerate(responses):
            print(f"\nRequest {i+1}:")
            print(f"Response: {response['choices'][0]['message']['content'][:100]}...")
            print(f"Token usage: {response['usage']}")
        
        print(f"\nTotal time for all requests: {end_time - start_time:.2f} seconds")
        
    finally:
        # Always close the client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

**Implementation insights:**
- Using `asyncio.gather()` to run multiple requests concurrently
- Measuring performance with parallel requests
- All requests start at the same time but may complete in different orders
- The client handles the concurrency and ensures all requests are processed

## Rate Limiting Implementation

### 3.1 Basic Rate Limit Handling Implementation

This example demonstrates implementing rate limit handling:

```python
import asyncio
import time
from CustomGroqChat import GroqClient

async def main():
    # Initialize the client
    client = GroqClient()
    await client.initialize()
    
    try:
        # Get available models
        models = client.get_available_models()
        if not models:
            print("No models available")
            return
            
        model_name = models[0]
        print(f"Using model: {model_name}")
        
        # Check initial limits
        limits = await client.check_model_limits(model_name)
        print(f"Initial limits - minute: {limits['minute_used']}/{limits['minute_limit']}, day: {limits['day_used']}/{limits['day_limit']}")
        
        # Send 8 requests with retry
        start_time = time.time()
        tasks = []
        
        for i in range(1, 9):
            messages = [
                {"role": "user", "content": f"What is {i}+{i}?"}
            ]
            
            task = client.chat_completion(
                model_name=model_name,
                messages=messages
            )
            tasks.append(task)
        
        # Wait for all to complete with retry
        responses = []
        for task in asyncio.as_completed(tasks):
            try:
                response = await task
                responses.append(response)
                print(f"Request completed: {response['choices'][0]['message']['content']}")
            except Exception as e:
                print(f"Request failed: {e}")
        
        end_time = time.time()
        
        # Check final limits
        limits = await client.check_model_limits(model_name)
        print(f"Final limits - minute: {limits['minute_used']}/{limits['minute_limit']}, day: {limits['day_used']}/{limits['day_limit']}")
        
        print(f"Completed {len(responses)} out of {len(tasks)} requests")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        
    finally:
        # Always close the client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

**Implementation insights:**
- Checking rate limits before and after a batch of requests
- Using `asyncio.as_completed()` to process responses as they arrive
- Implementing proper error handling for each request
- Tracking and measuring request completion

## See Also

- [Simple Usage Guide](simple_usage.md) - Step-by-step implementation instructions
- [Advanced Usage Guide](advanced_usage.md) - Advanced implementation techniques
- [Examples Overview](../examples.md) - Documentation for packaged example scripts 