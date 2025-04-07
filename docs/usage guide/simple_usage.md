# Simple Usage Guide

This guide will help you quickly implement the CustomGroqChat package in your application. We'll walk through each step from installation to creating your first working example.

## Step 1: Installation

Install the package using pip:

```bash
pip install customgroqchat
```

## Step 2: Configuration Setup

Create a `config.json` file in your project directory with the following structure:

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

> **Important**: Replace `"gsk_YOUR_GROQ_API_KEY"` with your actual GROQ API key.

### Configuration Fields Explained

| Field | Description | Recommended Value |
|-------|-------------|-------------------|
| `base_url` | GROQ API endpoint URL | `"https://api.groq.com/openai/v1"` |
| `api_key` | Your GROQ API key (starts with "gsk_") | Your actual API key |
| `req_per_minute` | Max requests per minute | 30 (default GROQ limit) |
| `req_per_day` | Max requests per day | 1200 or your plan limit |
| `token_per_minute` | Max tokens per minute | 7500 |
| `token_per_day` | Max tokens per day | 300000 or your plan limit |
| `context_window` | Model's max context size | 8192 for 8B models |

> **Tip**: You can add multiple models to your configuration by adding more objects with different model names as keys.

## Step 3: Create Your First Application

Copy and paste this minimal working example to get started:

```python
import asyncio
from CustomGroqChat import GroqClient

async def main():
    # 1. Initialize the client
    client = GroqClient("config.json")
    await client.initialize()
    
    try:
        # 2. Prepare your messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the capital of France?"}
        ]
        
        # 3. Generate a chat completion
        response = await client.chat_completion(
            model_name="llama-3.1-8b-instant",
            messages=messages
        )
        
        # 4. Extract and print the response
        content = response["choices"][0]["message"]["content"]
        print(f"Assistant: {content}")
        
        # 5. Access token usage information (optional)
        usage = response["usage"]
        print(f"\nToken usage: {usage['total_tokens']} total tokens")
        
    finally:
        # 6. Always close the client when done
        await client.close()

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
```

Save this as `simple_chat.py` and run it:

```bash
python simple_chat.py
```

## Step 4: Build a Multi-turn Conversation

For a more interactive experience, create a simple chat loop:

```python
import asyncio
from CustomGroqChat import GroqClient

async def interactive_chat():
    # Initialize the client
    client = GroqClient("config.json")
    await client.initialize()
    
    try:
        # Initialize message history with system prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        
        print("Chat with GROQ AI (type 'exit' to quit)")
        print("-----------------------------------------")
        
        # Chat loop
        while True:
            # Get user input
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                break
                
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Get response from AI
            print("Assistant is thinking...")
            response = await client.chat_completion(
                model_name="llama-3.1-8b-instant", 
                messages=messages
            )
            
            # Extract and display response
            assistant_message = response["choices"][0]["message"]["content"]
            print(f"\nAssistant: {assistant_message}")
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": assistant_message})
            
            # Show token usage (optional)
            usage = response["usage"]
            print(f"(Used {usage['total_tokens']} tokens)")
            
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(interactive_chat())
```

Save this as `interactive_chat.py` and run it:

```bash
python interactive_chat.py
```

## Step 5: Handle Errors Properly

Implement basic error handling in your application:

```python
import asyncio
from CustomGroqChat import GroqClient, CustomGroqChatException

async def robust_chat_example():
    try:
        # Initialize client (will look for config.json in current directory)
        client = GroqClient()
        await client.initialize()
        
        # Check which models are available
        models = client.get_available_models()
        if not models:
            print("No models found in configuration!")
            return
            
        print(f"Available models: {', '.join(models)}")
        model_name = models[0]  # Use the first model
        
        # Set up messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a short joke."}
        ]
        
        # Generate response
        response = await client.chat_completion(
            model_name=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        
        # Process response
        content = response["choices"][0]["message"]["content"]
        print(f"\nResponse from {model_name}:\n{content}")
        
    except FileNotFoundError:
        print("Error: Configuration file not found!")
        print("Make sure config.json exists in your project directory.")
    except CustomGroqChatException as e:
        print(f"Error during API call: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Clean up resources even if there was an error
        if 'client' in locals():
            await client.close()

if __name__ == "__main__":
    asyncio.run(robust_chat_example())
```

## Common Implementation Patterns

### Use Environment Variables for Configuration

For better security, you can use environment variables:

```python
import os
import json
from CustomGroqChat import GroqClient

# Set environment variable for API key
os.environ["GROQ_API_KEY"] = "gsk_your_api_key_here"

# Load config but use environment variable for API key
with open("config.json", "r") as f:
    config = json.load(f)

# Replace API key with environment variable
for model in config:
    config[model]["api_key"] = os.environ["GROQ_API_KEY"]

# Create client with the modified config
client = GroqClient(config_data=config)  # Pass config directly
```

### Check Rate Limits Before Making Requests

To monitor your API usage:

```python
# Check current rate limit status
limits = await client.check_model_limits("llama-3.1-8b-instant")
print(f"Minute usage: {limits['minute_used']}/{limits['minute_limit']}")
print(f"Day usage: {limits['day_used']}/{limits['day_limit']}")
```

## Next Steps

After mastering these basics, you can:

1. Explore [advanced usage features](advanced_usage.md) like request prioritization
2. Learn about [handling exceptions](exceptions.md) in more detail
3. Check out complete [example applications](../examples.md) demonstrating real-world usage

For a deeper understanding of how the code works, see our [detailed code explanations](../example/examples_explained.md). 