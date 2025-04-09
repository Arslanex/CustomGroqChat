# CustomGroqChat

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![Status](https://img.shields.io/badge/status-stable-green.svg)

A powerful Python client for the Groq Cloud API with intelligent rate limiting, request prioritization, token counting, and comprehensive error handling.

## Project Overview

CustomGroqChat is designed to simplify interaction with Groq's Large Language Models while providing enterprise-grade reliability features. It addresses common challenges when working with LLM APIs:

- **Rate limit management** - Automatically handles API quota limits without errors
- **Request prioritization** - Ensures critical requests get processed first
- **Parallel processing** - Efficiently manages concurrent requests
- **Error resilience** - Gracefully handles network issues and API errors
- **Token optimization** - Tracks and manages token usage for cost control

Whether you're building a chatbot, content generation system, or AI-powered workflow, CustomGroqChat provides the infrastructure to make your application reliable, efficient, and scalable.

## Documentation

| Section | Description |
|---------|-------------|
| [Overview](docs/overview.md) | Introduction to CustomGroqChat |
| [Usage Guides](docs/usage%20guide/) | Step-by-step guides for using the package |
| [Examples](docs/example/) | Practical examples and use cases |
| [API Reference](docs/scripts/) | Detailed documentation for each component |
| [Error Handling](docs/usage%20guide/exceptions.md) | Guide to handling errors |

## System Architecture

CustomGroqChat uses a layered architecture designed for flexibility and robustness:

```
┌─────────────────────────────────────────┐
│ Application Layer (GroqClient)          │
├─────────────────────────────────────────┤
│ Request Management (RequestHandler)     │
├─────────────────────────────────────────┤
│ Concurrency (QueueManager)              │
├─────────────────────────────────────────┤
│ Rate Management (RateLimitHandler)      │
├─────────────────────────────────────────┤
│ API Communication (APIClient)           │
└─────────────────────────────────────────┘
```

## Features

### Rate Limiting & Quota Management

- **Intelligent Rate Limiting**: 
  - Automatic management of API rate limits to prevent quota errors
  - Separate tracking for requests and tokens usage
  - Sliding window implementation for optimal throughput
  - Per-model rate limit configuration
  
- **Token Budget Management**:
  - Real-time token usage tracking and limits
  - Daily and per-minute quota enforcement
  - Predictive token estimation before requests
  - Usage reporting and budget allocation

### Request Processing

- **Priority-based Request Queuing**: 
  - Process critical requests ahead of less important ones
  - Three priority levels: high, normal, and low
  - Fair scheduling to prevent starvation of low-priority requests
  - Dynamic priority adjustment based on wait time
  
- **Parallel Request Processing**: 
  - Handle multiple concurrent requests efficiently
  - Automatic request batching for better throughput
  - Progress tracking for long-running operations
  - Optimized resource utilization

### API Communication

- **Multiple Model Support**: 
  - Use different Groq models with model-specific settings
  - Easy switching between models at runtime
  - Automatic model capability detection
  - Configuration for model-specific parameters
  
- **Asynchronous API**: 
  - Built with asyncio for efficient concurrent processing
  - Non-blocking request handling
  - Callback support for event-driven architectures
  - Streaming response support

### Error Handling & Resilience

- **Comprehensive Error Handling**: 
  - Detailed exception hierarchy for better error management
  - Automatic retries for transient errors with exponential backoff
  - Detailed error information for debugging
  - Graceful degradation options
  
- **Token Counting**: 
  - Accurate token usage estimation for context window management
  - Support for different tokenizers per model
  - Prediction of completion token usage
  - Context window optimization

### Developer Experience

- **Easy Configuration**:
  - Simple JSON configuration
  - Environment variable support
  - Per-model settings
  - Reasonable defaults

- **Extensive Logging**:
  - Detailed debug information
  - Performance metrics
  - Usage statistics
  - Request tracing

## Example Usage

### Configuration

Create a `config.json` file in your project directory:

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
  "llama-3.1-70b-instant": {
    "base_url": "https://api.groq.com/openai/v1",
    "api_key": "YOUR_GROQ_API_KEY",
    "req_per_minute": 15,
    "req_per_day": 600,
    "token_per_minute": 3750,
    "token_per_day": 150000,
    "context_window": 8192
  }
}
```

### Basic Chat

```python
import asyncio
from CustomGroqChat import GroqClient

async def main():
    # Initialize the client with default configuration
    client = GroqClient()
    await client.initialize()
    
    try:
        # ======== SEND A REQUEST ========
        response = await client.chat_completion(
            model_name="llama-3.1-8b-instant",  # Choose your preferred model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a short joke."}
            ],
            temperature=0.7,  # Control randomness (0.0 to 1.0)
            max_tokens=100    # Limit response length
        )
        
        # Extract and display the response content
        assistant_message = response["choices"][0]["message"]["content"]
        print("\n" + "=" * 50)
        print("ASSISTANT RESPONSE:")
        print("=" * 50)
        print(assistant_message)
        print("=" * 50 + "\n")
        
    finally:
        # Always close the client to clean up resources
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Choose Model

```python
import asyncio
from CustomGroqChat import GroqClient

async def main():
    client = GroqClient()
    await client.initialize()
    
    try:
        # Get available models
        models = await client.get_available_models()
        print(f"Available models: {', '.join(models)}")
        
        # Select a model based on your needs
        selected_model = "llama-3.1-8b-instant"  # Fast responses
        # selected_model = "llama-3.1-70b-instant"  # Higher quality
        
        print(f"Using model: {selected_model}")
        
        # Send request to the chosen model
        response = await client.chat_completion(
            model_name=selected_model,
            messages=[{"role": "user", "content": "Explain what makes a good API client."}],
            max_tokens=150
        )
        
        print("\nRESPONSE:")
        print("-" * 40)
        print(response["choices"][0]["message"]["content"])
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Parallel Requests

```python
import asyncio
from CustomGroqChat import GroqClient

async def main():
    client = GroqClient()
    await client.initialize()
    
    try:
        print("Sending multiple parallel requests...")
        
        # ======== CREATE MULTIPLE REQUESTS ========
        tasks = []
        questions = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Give me a recipe for chocolate cake.",
            "What are the benefits of exercise?",
            "Write a haiku about programming."
        ]
        
        # Create tasks with different priorities
        for i, question in enumerate(questions):
            # Determine priority based on question index
            if i == 0:
                priority = "high"      # First question gets high priority
            elif i == len(questions)-1:
                priority = "low"       # Last question gets low priority
            else:
                priority = "normal"    # Others get normal priority
                
            print(f"Queueing question {i+1} with {priority} priority: {question[:30]}...")
            
            # Create task (not awaited yet)
            task = client.chat_completion(
                model_name="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": question}],
                priority=priority,
                max_tokens=150
            )
            tasks.append(task)
        
        # Process all requests concurrently
        responses = await asyncio.gather(*tasks)
        
        # Process responses
        for i, response in enumerate(responses):
            print(f"\nRESPONSE {i+1}: {questions[i][:30]}...")
            print("-" * 40)
            print(response["choices"][0]["message"]["content"][:100] + "...")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Example Scripts

| Script | Description |
|--------|-------------|
| `examples/1_compare_all_models.py` | Compare performance across different Groq models |
| `examples/2_select_model_and_chat.py` | Interactive model selection and chat |
| `examples/3_handle_rate_limits.py` | Demonstrates rate limit handling |
| `examples/3_rate_limit_test.py` | Tests rate limit handling with moderate load |
| `examples/3_rate_limit_test_aggressive.py` | Tests rate limit handling with heavy load |
| `examples/3_rate_limit_test_exceed.py` | Deliberately exceeds rate limits to test recovery |
| `examples/4_conversation_with_memory.py` | Multi-turn conversations with context management |
| `examples/5_parallel_processing.py` | Process multiple requests in parallel with callbacks |

## Performance Tips

- **Reuse client instances**: Create a single GroqClient for your application to optimize rate limit handling
- **Set appropriate priorities**: Use "high" priority only for time-sensitive requests
- **Batch similar requests**: Group related requests together for better throughput
- **Use smaller models for simpler tasks**: Switch to smaller, faster models when appropriate
- **Implement request caching**: Cache responses for identical requests to save tokens and time
- **Set reasonable token limits**: Limit max_tokens to avoid unnecessarily large responses
- **Monitor token usage**: Use the token counting features to track and optimize costs

## Troubleshooting

### Common Issues

1. **Rate Limits Exceeded**
   - Check your config.json rate limit settings match your actual API limits
   - Implement exponential backoff for retries
   - Consider upgrading your Groq API tier

2. **High Latency**
   - Reduce the number of concurrent requests
   - Check network connectivity
   - Use a smaller model for faster responses

3. **Memory Issues**
   - Limit conversation history length
   - Close client instances when done
   - Process responses in batches

## FAQ

**Q: Can I use CustomGroqChat with other LLM providers?**  
A: CustomGroqChat is specifically designed for the Groq API, but the architecture could be adapted for other providers with similar APIs.

**Q: How do I handle streaming responses?**  
A: Streaming is supported through the `stream=True` parameter in chat_completion requests. Use callback functions to process each chunk as it arrives.

**Q: What's the recommended way to handle rate limits?**  
A: CustomGroqChat handles rate limits automatically, but you can customize behavior by adjusting the configuration and implementing callbacks for waiting periods.

**Q: Is CustomGroqChat suitable for production use?**  
A: Yes, CustomGroqChat is designed for production environments with features like error handling, rate limit management, and queue prioritization.

**Q: How do I contribute to the project?**  
A: See the Contributing section below. We welcome bug reports, feature requests, and pull requests.

## Compatibility

- **Python Versions**: 3.8+
- **Operating Systems**: Windows, macOS, Linux
- **Dependencies**: asyncio, aiohttp, tiktoken, json, logging
- **Groq API Versions**: v1

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feature: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Groq team for their excellent API
- All contributors to this project

---

Built with ❤️ for the AI community
