# CustomGroqChat Overview

CustomGroqChat is a powerful Python client for interacting with the GROQ Cloud API. It provides a straightforward interface to generate text and chat completions while intelligently handling rate limiting, token management, and request queuing.

## Table of Contents

1. [Implementation Guide](usage%20guide/user_guide.md)
2. [Simple Usage](usage%20guide/simple_usage.md)
3. [Advanced Usage](usage%20guide/advanced_usage.md)
4. [Exceptions](usage%20guide/exceptions.md)
5. [Examples Overview](examples.md)

## Key Features

CustomGroqChat provides these key features:

- **Asynchronous API** for optimal performance
- **Intelligent rate limiting** that automatically paces requests to avoid API limits
- **Priority-based request queuing** for handling multiple concurrent requests
- **Accurate token counting** to manage context windows
- **Automatic error handling and retries**
- **Simple configuration management**
- **Comprehensive exception handling**
- **Multiple model support** with individual rate limit configurations

The package is designed to make it easy to interact with GROQ's models without having to worry about the underlying API details or rate limits.

## Key Insights

Based on extensive testing, here are some key insights about how CustomGroqChat works:

- **Self-regulating rate limiting**: Even when you try to exceed API rate limits, the package intelligently queues and paces requests to avoid failures.
- **Automatic request distribution**: When sending multiple concurrent requests, the library automatically spreads them out over time to stay within rate limits.
- **Graceful error handling**: The package will automatically retry requests when rate limits are hit, and provide clear error messages for other types of failures.
- **Configurable models**: You can easily configure and use different GROQ models with different rate limits in a single application.
- **Prioritization system**: Critical requests can be assigned higher priority (lower numbers) to be processed before less important ones.

## Supported Models

CustomGroqChat works with all GROQ models, including:

- **Llama 3.1 Series**: llama-3.1-8b-instant, llama-3.1-70b-instant, llama-3.1-405b-instant
- **Gemma 2 Series**: gemma2-9b-it, gemma2-27b-it
- **Mixtral Series**: mixtral-8x7b-instant
- Any new models added to the GROQ platform

## Before You Begin

To use CustomGroqChat, you'll need:

1. A GROQ API key (sign up at [groq.com](https://groq.com))
2. A configuration file with your API settings
3. Python 3.7+ with asyncio support

## Configuration File

Create a `config.json` file with your models and their limits:

```json
{
  "llama-3.1-8b-instant": {
    "base_url": "https://api.groq.com/openai/v1",
    "api_key": "YOUR_GROQ_API_KEY",
    "req_per_minute": 30,
    "req_per_day": 1200,
    "token_per_minute": 7500,
    "token_per_day": 300000,
    "context_window": 8192
  },
  "gemma2-9b-it": {
    "base_url": "https://api.groq.com/openai/v1",
    "api_key": "YOUR_GROQ_API_KEY",
    "req_per_minute": 10,
    "req_per_day": 700,
    "token_per_minute": 4500,
    "token_per_day": 200000,
    "context_window": 32768
  }
}
```

## Getting Started

For implementation instructions and examples, please refer to:

- [Implementation Guide](usage%20guide/user_guide.md) - Step-by-step implementation instructions
- [Simple Usage](usage%20guide/simple_usage.md) - Basic setup and common operations
- [Advanced Usage](usage%20guide/advanced_usage.md) - Advanced features and techniques
- [Exceptions](usage%20guide/exceptions.md) - Comprehensive error handling documentation
- [Examples Overview](examples.md) - Example scripts and best practices 