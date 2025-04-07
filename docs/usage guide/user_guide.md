# CustomGroqChat Implementation Guide

This guide provides a practical overview of implementing CustomGroqChat in your Python applications. You'll learn how to set up, configure, and use the key features of this powerful GROQ API client.

## Implementation Overview

CustomGroqChat is designed to simplify your integration with GROQ's AI models while providing advanced features:

1. **Easy Implementation**: Set up in just a few lines of code
2. **Automatic Rate Limit Management**: No need to implement complex rate-limiting logic
3. **Priority-Based Queuing**: Control which requests are processed first
4. **Accurate Token Counting**: Manage context windows and costs effectively
5. **Async Support**: Handle concurrent requests efficiently
6. **Error Handling**: Comprehensive exception system for robust applications

## Implementation Steps

### Step 1: Installation

```bash
pip install customgroqchat
```

### Step 2: Create Your Configuration

Create a `config.json` file with your GROQ API key and model settings:

```json
{
  "llama-3.1-8b-instant": {
    "base_url": "https://api.groq.com/openai/v1",
    "api_key": "gsk_YOUR_GROQ_API_KEY",
    "req_per_minute": 30,
    "req_per_day": 1200,
    "token_per_minute": 7500,
    "token_per_day": 300000,
    "context_window": 8192
  }
}
```

### Step 3: Initialize the Client

```python
import asyncio
from CustomGroqChat import GroqClient

async def main():
    # Initialize the client with your config file
    client = GroqClient("config.json")
    await client.initialize()
    
    # Your code here...
    
    # Always close the client when done
    await client.close()

# Run your main function
asyncio.run(main())
```

### Step 4: Make Your First API Call

```python
# Inside your main function:
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
]

response = await client.chat_completion(
    model_name="llama-3.1-8b-instant",
    messages=messages
)

# Extract and use the response
content = response["choices"][0]["message"]["content"]
print(f"Assistant: {content}")
```

### Step 5: Handle Errors Properly

```python
from CustomGroqChat import CustomGroqChatException

try:
    response = await client.chat_completion(
        model_name="llama-3.1-8b-instant",
        messages=messages
    )
    
    # Process response
    print(response["choices"][0]["message"]["content"])
    
except CustomGroqChatException as e:
    print(f"Error: {e}")
finally:
    await client.close()
```

## Common Implementation Patterns

### Checking Available Models

```python
# Get all models from your configuration
models = client.get_available_models()
print(f"Available models: {models}")
```

### Creating a Multi-Turn Conversation

```python
# Initialize conversation history
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

# First user turn
messages.append({"role": "user", "content": "Hi there!"})
response = await client.chat_completion(
    model_name="llama-3.1-8b-instant",
    messages=messages
)

# Add assistant response to history
assistant_message = response["choices"][0]["message"]["content"]
messages.append({"role": "assistant", "content": assistant_message})

# Second user turn (with conversation context preserved)
messages.append({"role": "user", "content": "What can you help me with today?"})
# Continue the conversation...
```

## Next Implementation Steps

After mastering the basics, explore these advanced implementation patterns:

1. **Rate Limiting**: See [Advanced Usage - Rate Limiting](advanced_usage.md#rate-limiting)
2. **Request Prioritization**: See [Advanced Usage - Request Prioritization](advanced_usage.md#request-prioritization)
3. **Token Management**: See [Advanced Usage - Token Counting](advanced_usage.md#token-counting-and-budget-management)
4. **Error Handling**: See [Exceptions Guide](exceptions.md)
5. **Parallel Processing**: See [Advanced Usage - Parallel Processing](advanced_usage.md#parallel-request-processing)

## Complete Implementation Example

Here's a complete implementation example you can adapt for your application:

```python
import asyncio
import os
from CustomGroqChat import GroqClient, CustomGroqChatException

async def main():
    # Setup: Use environment variable for API key for better security
    os.environ["GROQ_API_KEY"] = "gsk_your_api_key_here"
    
    try:
        # Initialization
        client = GroqClient("config.json")
        await client.initialize()
        
        # Get available models
        models = client.get_available_models()
        if not models:
            print("No models found. Check your configuration.")
            return
            
        model_name = models[0]  # Use first available model
        print(f"Using model: {model_name}")
        
        # Basic conversation setup
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Can you help me learn about quantum computing?"}
        ]
        
        # Generate response
        print("Generating response...")
        response = await client.chat_completion(
            model_name=model_name,
            messages=messages
        )
        
        # Extract and display response
        content = response["choices"][0]["message"]["content"]
        print(f"\nAssistant: {content}")
        
        # Show token usage
        usage = response["usage"]
        print(f"\nToken usage: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total")
        
    except FileNotFoundError:
        print("Configuration file not found.")
    except CustomGroqChatException as e:
        print(f"API error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if 'client' in locals():
            await client.close()
            print("Client closed.")

if __name__ == "__main__":
    asyncio.run(main())
```

For more detailed implementation examples, check out the [Simple Usage Guide](simple_usage.md) and [Examples Documentation](../examples.md). 