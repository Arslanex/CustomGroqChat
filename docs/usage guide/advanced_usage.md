# Advanced Usage Guide

This guide covers advanced features of the CustomGroqChat package with implementation examples for each feature. Learn how to optimize your application with rate limiting, request prioritization, token counting, and more.

## Table of Contents
- [Rate Limiting](#rate-limiting)
- [Request Prioritization](#request-prioritization)
- [Token Counting and Budget Management](#token-counting-and-budget-management)
- [Parallel Request Processing](#parallel-request-processing)
- [Using Callbacks](#using-callbacks)
- [Managing Multiple Models](#managing-multiple-models)

## Rate Limiting

CustomGroqChat includes intelligent rate limiting that prevents API errors by automatically managing your request frequency. Here's how to implement and leverage this feature:

### Implementation Example

```python
import asyncio
import time
from CustomGroqChat import GroqClient

async def rate_limit_example():
    client = GroqClient("config.json")
    await client.initialize()
    
    try:
        # Check initial rate limit status
        model_name = "llama-3.1-8b-instant"
        limits = await client.check_model_limits(model_name)
        print(f"Initial limits - minute: {limits['minute_used']}/{limits['minute_limit']}, day: {limits['day_used']}/{limits['day_limit']}")
        
        # Send multiple requests that would normally exceed rate limits
        print("\nSending 35 requests (exceeds standard 30 req/min limit)...")
        start_time = time.time()
        tasks = []
        
        # Create 35 simple requests
        for i in range(1, 36):
            messages = [
                {"role": "user", "content": f"What is {i}+{i}?"}
            ]
            
            task = client.chat_completion(
                model_name=model_name,
                messages=messages
            )
            tasks.append(task)
        
        # Wait for all to complete
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Print results
        print(f"\nAll {len(responses)} requests completed successfully in {end_time - start_time:.2f} seconds")
        
        # Check final rate limit status
        limits = await client.check_model_limits(model_name)
        print(f"Final limits - minute: {limits['minute_used']}/{limits['minute_limit']}, day: {limits['day_used']}/{limits['day_limit']}")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(rate_limit_example())
```

### How It Works

1. CustomGroqChat maintains an internal queue of requests
2. When requests exceed the configured rate limit, they're automatically delayed
3. The client distributes requests across time windows to stay within limits
4. No need to manually implement retries or delays

### Configuring Custom Rate Limits

You can adjust the rate limits in your configuration file:

```json
{
  "llama-3.1-8b-instant": {
    "base_url": "https://api.groq.com/openai/v1",
    "api_key": "gsk_YOUR_GROQ_API_KEY",
    "req_per_minute": 20,  // Lower than default for safety
    "req_per_day": 1000,
    "token_per_minute": 6000,
    "token_per_day": 250000,
    "context_window": 8192
  }
}
```

> **Tip**: Setting limits below GROQ's actual limits adds a safety margin and ensures your application will continue functioning even if GROQ's limits change.

## Request Prioritization

CustomGroqChat allows you to prioritize important requests over less critical ones. Here's how to implement it:

### Implementation Example

```python
import asyncio
import time
from CustomGroqChat import GroqClient

async def priority_example():
    client = GroqClient("config.json")
    await client.initialize()
    
    try:
        model_name = "llama-3.1-8b-instant"
        start_time = time.time()
        tasks = []
        
        # Add a high priority request (will be processed first)
        print("Adding high priority request...")
        messages = [{"role": "user", "content": "URGENT: What is the current time in New York?"}]
        high_task = client.chat_completion(
            model_name=model_name,
            messages=messages,
            priority=1  # Lower number = higher priority (1-10)
        )
        tasks.append(high_task)
        
        # Add 5 low priority requests
        print("Adding 5 low priority requests...")
        for i in range(1, 6):
            messages = [{"role": "user", "content": f"Low priority question {i}: What is {i} + {i}?"}]
            task = client.chat_completion(
                model_name=model_name,
                messages=messages,
                priority=10  # Lowest priority
            )
            tasks.append(task)
            
        # Add a medium priority request
        print("Adding medium priority request...")
        messages = [{"role": "user", "content": "Medium priority: What is the capital of Spain?"}]
        medium_task = client.chat_completion(
            model_name=model_name,
            messages=messages,
            priority=5  # Medium priority
        )
        tasks.append(medium_task)
        
        # Process responses as they arrive
        for task in asyncio.as_completed(tasks):
            response = await task
            content = response["choices"][0]["message"]["content"]
            # We determine which request this is by looking at the content
            elapsed = time.time() - start_time
            priority_level = "high" if "URGENT" in content else "medium" if "Spain" in content else "low"
            print(f"[{elapsed:.2f}s] Completed {priority_level} priority request: {content[:50]}...")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(priority_example())
```

### Priority Levels

- `priority=1`: Highest priority (critical requests)
- `priority=5`: Medium priority (default)
- `priority=10`: Lowest priority (background tasks)

### When to Use Prioritization

- User-facing interactions (high priority)
- Background batch processing (low priority)
- System maintenance tasks (low priority)
```python
import asyncio
from CustomGroqChat import GroqClient

async def process_response(response, metadata):
    question_number = metadata["question_number"]
    answer = response["choices"][0]["message"]["content"]
    print(f"Question {question_number}: {answer}")

async def callbacks_example():
    client = GroqClient()
    await client.initialize()
    
    try:
        # Queue several requests with callbacks
        tasks = []
        for i in range(1, 11):
            messages = [{"role": "user", "content": f"What is {i}×{i}?"}]
            
            # Pass both the callback and metadata
            task = client.chat_completion(
                model_name="llama-3.1-8b-instant",
                messages=messages,
                callback=process_response,
                callback_metadata={"question_number": i}
            )
            tasks.append(task)
        
        # Wait for all to complete
        await asyncio.gather(*tasks)
        
    finally:
        await client.close()
```

## Token Counting and Budget Management

CustomGroqChat provides tools to estimate token usage before sending requests:

```python
import asyncio
from CustomGroqChat import GroqClient

async def token_counting_example():
    client = GroqClient()
    await client.initialize()
    
    try:
        # Prepare a message
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short poem about artificial intelligence."}
        ]
        
        # Count tokens before sending
        estimated_tokens = await client.count_tokens(
            model_name="llama-3.1-8b-instant", 
            messages=messages
        )
        print(f"Estimated prompt tokens: {estimated_tokens}")
        
        # Check if within budget
        max_budget = 100  # tokens
        if estimated_tokens > max_budget:
            print(f"Request exceeds token budget of {max_budget}")
        else:
            # Proceed with request
            response = await client.chat_completion(
                model_name="llama-3.1-8b-instant",
                messages=messages
            )
            
            # Get actual token usage
            actual_usage = response["usage"]
            print(f"Actual prompt tokens: {actual_usage['prompt_tokens']}")
            print(f"Completion tokens: {actual_usage['completion_tokens']}")
            print(f"Total tokens: {actual_usage['total_tokens']}")
        
    finally:
        await client.close()
```

## Request Cancellation

You can cancel in-flight requests if they're no longer needed:

```python
import asyncio
from CustomGroqChat import GroqClient

async def cancellation_example():
    client = GroqClient()
    await client.initialize()
    
    try:
        # Start a request
        messages = [{"role": "user", "content": "Write a very detailed essay about quantum physics."}]
        task = asyncio.create_task(
            client.chat_completion(
                model_name="llama-3.1-8b-instant",
                messages=messages
            )
        )
        
        # Some condition causes us to cancel the request
        await asyncio.sleep(0.5)  # Simulating some processing time
        print("User requested cancellation")
        
        # Cancel the request
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            print("Request was cancelled successfully")
            
    finally:
        await client.close()
```

## Retrying Failures

For handling transient errors, you can implement retry logic:

```python
import asyncio
from CustomGroqChat import GroqClient, CustomGroqChatException

async def retry_example():
    client = GroqClient()
    await client.initialize()
    
    try:
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                messages = [{"role": "user", "content": "Summarize the theory of relativity"}]
                response = await client.chat_completion(
                    model_name="llama-3.1-8b-instant",
                    messages=messages
                )
                
                # If successful, break the loop
                print(response["choices"][0]["message"]["content"])
                break
                
            except CustomGroqChatException as e:
                retry_count += 1
                if retry_count < max_retries:
                    # Exponential backoff
                    wait_time = 2 ** retry_count
                    print(f"Error: {e}. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts. Error: {e}")
                    
    finally:
        await client.close()
```

## Configuration Reloading

You can reload the configuration at runtime if you need to update API keys or rate limits:

```python
import asyncio
from CustomGroqChat import GroqClient

async def reload_config_example():
    client = GroqClient("config.json")
    await client.initialize()
    
    try:
        # Use the client with initial configuration
        response1 = await client.chat_completion(
            model_name="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print("First response received")
        
        # Reload configuration (perhaps after editing the config file)
        await client.reload_config()
        print("Configuration reloaded")
        
        # Continue using the client with updated configuration
        response2 = await client.chat_completion(
            model_name="llama-3.1-8b-instant", 
            messages=[{"role": "user", "content": "What's new?"}]
        )
        print("Second response received with updated configuration")
        
    finally:
        await client.close()
```

## Combining Advanced Features

You can combine multiple advanced features for complex scenarios:

```python
import asyncio
import time
from CustomGroqChat import GroqClient

async def comprehensive_example():
    client = GroqClient()
    await client.initialize()
    
    try:
        # Check available resources
        limits = await client.check_model_limits("llama-3.1-8b-instant")
        print(f"Available capacity: {limits['minute_limit'] - limits['minute_used']} requests this minute")
        
        # Prepare different priority requests
        high_priority_messages = [{"role": "user", "content": "URGENT: Who won the last World Cup?"}]
        medium_priority_messages = [{"role": "user", "content": "When is the next solar eclipse?"}]
        low_priority_messages = []
        
        for i in range(1, 11):
            low_priority_messages.append([{"role": "user", "content": f"Tell me a fun fact about the number {i}"}])
        
        # Submit requests with different priorities
        tasks = []
        
        # High priority request
        high_task = client.chat_completion(
            model_name="llama-3.1-8b-instant",
            messages=high_priority_messages,
            priority=1
        )
        tasks.append(high_task)
        
        # Medium priority request
        medium_task = client.chat_completion(
            model_name="llama-3.1-8b-instant",
            messages=medium_priority_messages,
            priority=5
        )
        tasks.append(medium_task)
        
        # Low priority requests
        for messages in low_priority_messages:
            low_task = client.chat_completion(
                model_name="llama-3.1-8b-instant",
                messages=messages,
                priority=10
            )
            tasks.append(low_task)
        
        # Start timer
        start_time = time.time()
        
        # Process responses as they complete
        for completed_task in asyncio.as_completed(tasks):
            response = await completed_task
            elapsed = time.time() - start_time
            content = response["choices"][0]["message"]["content"]
            print(f"[{elapsed:.2f}s] Response: {content[:50]}...")
        
        # Check final usage
        final_limits = await client.check_model_limits("llama-3.1-8b-instant")
        print(f"Final usage: {final_limits['minute_used']}/{final_limits['minute_limit']} requests this minute")
        
    finally:
        await client.close()
```

## Best Practices

1. **Always initialize and close the client**: Use `await client.initialize()` before making requests and `await client.close()` when finished.

2. **Use priority wisely**: Reserve higher priorities (lower numbers) for truly urgent requests.

3. **Handle errors gracefully**: Catch `CustomGroqChatException` for proper error handling.

4. **Check queue status**: Monitor rate limits with `check_model_limits` to avoid surprises.

5. **Use async patterns**: Leverage `asyncio.gather()` for multiple requests and `asyncio.as_completed()` when order doesn't matter.

6. **Estimate token usage**: Use `count_tokens` to estimate costs and stay within context windows.

7. **Clean up resources**: Always close the client, even after errors, using try/finally blocks.

8. **Use callbacks for non-blocking operations**: Especially important for long-running applications. 