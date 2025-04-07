# CustomGroqChat Implementation Guide

This directory contains comprehensive guides to help you quickly implement and use the CustomGroqChat package in your applications.

## Quick Start

If you're in a hurry, here's how to get started in 3 steps:

1. **Install the package**:
   ```bash
   pip install customgroqchat
   ```

2. **Create a minimal config.json**:
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

3. **Run your first query**:
   ```python
   import asyncio
   from CustomGroqChat import GroqClient

   async def main():
       client = GroqClient("config.json")
       await client.initialize()
       
       try:
           response = await client.chat_completion(
               model_name="llama-3.1-8b-instant",
               messages=[{"role": "user", "content": "Hello, how are you?"}]
           )
           print(response["choices"][0]["message"]["content"])
       finally:
           await client.close()

   asyncio.run(main())
   ```

For detailed implementation instructions, follow the guides below.

## Available Guides

### Core Implementation Guides

- [**Implementation Guide**](user_guide.md) - Step-by-step implementation instructions with copy-paste examples
- [**Simple Usage Guide**](simple_usage.md) - Step-by-step implementation instructions with copy-paste examples
- [**Advanced Usage Guide**](advanced_usage.md) - Implementation patterns for rate limiting, prioritization and more
- [**Exceptions Guide**](exceptions.md) - How to implement proper error handling in your application
- [**Implementation Examples**](implementation_examples.md) - Detailed code examples for various implementation patterns

### Content By Implementation Stage

#### 1. Getting Started (Beginner)
The [**Simple Usage Guide**](simple_usage.md) covers all the basics you need to implement a working application:
- Setting up your environment and configuration
- Building your first application
- Creating interactive conversations
- Adding basic error handling

#### 2. Optimizing Your Application (Intermediate)
The [**Advanced Usage Guide**](advanced_usage.md) covers techniques to enhance your implementation:
- Implementing efficient rate limiting
- Adding request prioritization for better UX
- Managing token budgets and counting
- Processing multiple requests in parallel
- Using callbacks for non-blocking operations

#### 3. Production-Ready Code (Advanced)
The [**Exceptions Guide**](exceptions.md) and later sections of the Advanced guide help you build robust applications:
- Implementing comprehensive error handling
- Creating resilient retry mechanisms
- Managing multiple models efficiently
- Designing scalable applications

## Learning Path

We recommend this implementation order:

1. **Day 1**: Follow the [Simple Usage Guide](simple_usage.md) to get your first application running
2. **Day 2**: Add more robust error handling using the [Exceptions Guide](exceptions.md)
3. **Day 3**: Optimize your application with the [Advanced Usage Guide](advanced_usage.md)
4. **Day 4+**: Explore the [example applications](../examples.md) for real-world patterns

## Code Examples

All guides include copy-paste code snippets that you can immediately use in your application. For complete implementation examples, see the [Examples Directory](../example/).

## Implementation Support

If you encounter issues while implementing CustomGroqChat:

1. Check the [Exceptions Guide](exceptions.md) for common error solutions
2. Review the [detailed code explanations](../example/examples_explained.md) for implementation patterns
3. For additional help, create an issue on the project's GitHub repository

## See Also

- [Examples](../examples.md) - Complete example applications
- [Scripts Documentation](../scripts/) - Internal components documentation
- [Tests Documentation](../tests/) - Test suite coverage
