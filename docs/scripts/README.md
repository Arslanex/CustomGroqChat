# CustomGroqChat Scripts Documentation

This directory contains documentation for the internal components (scripts) of the CustomGroqChat package.

## Component Architecture

The CustomGroqChat package is built with a modular architecture, where each component handles a specific responsibility. This design enables flexible customization, extensibility, and easier maintenance.

### Architecture Diagram

```
                       ┌─────────────┐
                       │ GroqClient  │
                       └─────┬───────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ RequestHandler  │
                    └────┬─────┬──────┘
                         │     │
           ┌─────────────┘     └──────────────┐
           │                                  │
┌──────────▼─────────┐              ┌─────────▼────────┐
│   QueueManager     │              │  TokenCounter    │
└──────────┬─────────┘              └──────────────────┘
           │
           │
┌──────────▼─────────┐     ┌────────────────────┐
│  RateLimitHandler  │◄────┤    ConfigLoader    │
└──────────┬─────────┘     └────────────────────┘
           │
┌──────────▼─────────┐
│     APIClient      │
└────────────────────┘
```

## Available Components

| Component | Documentation |
|-----------|---------------|
| **Package Overview** | [Package Exports](package_exports.md) |
| **Main Interface** | [GroqClient](groq_client.md) |
| **Request Processing** | [RequestHandler](request_handler.md) |
| **Queue Management** | [QueueManager](queue_manager.md) |
| **API Communication** | [APIClient](api_client.md) |
| **Rate Limiting** | [RateLimitHandler](rate_limit_handler.md) |
| **Configuration** | [ConfigLoader](config_loader.md) |
| **Token Counting** | [TokenCounter](token_counter.md) |
| **Error Handling** | [Exceptions](exceptions.md) |

## Component Descriptions

### Main Interface

- **GroqClient**: The primary interface for applications integrating with the Groq Cloud API. It provides high-level methods for chat completions, text completions, and other API features, abstracting away the complexity of request handling, queuing, and rate limiting.

### Request Processing

- **RequestHandler**: Validates and prepares requests before they're sent to the API. Manages token counting, parameter validation, and converts application-level requests to API-compatible formats.

### Queue Management

- **QueueManager**: Implements a priority queue system for handling concurrent requests efficiently. Maintains separate queues for different priority levels and ensures requests are processed in the appropriate order.

### API Communication

- **APIClient**: Handles direct communication with the Groq Cloud API. Manages connection pooling, timeout handling, and response parsing. Abstracts the HTTP layer, providing a clean interface for other components.

### Utilities

- **RateLimitHandler**: Tracks and enforces rate limits for different models. Implements token bucket algorithms to handle both per-minute and per-day limits efficiently.

- **ConfigLoader**: Manages configuration loading from files or environment variables. Validates configuration values and provides sensible defaults.

- **TokenCounter**: Accurately counts tokens for different request types. Ensures requests stay within model context limits and helps track token-based rate limits.

- **Exceptions**: Defines the custom exception hierarchy used throughout the package, providing detailed error information and handling guidance.

## Component Interactions

- The `GroqClient` uses the `RequestHandler` to prepare and validate requests.
- The `RequestHandler` uses the `TokenCounter` to validate token limits and the `QueueManager` to queue requests.
- The `QueueManager` uses the `RateLimitHandler` to check rate limits and the `APIClient` to send requests.
- The `RateLimitHandler` uses the `ConfigLoader` to get rate limit configurations.

## Implementation Details

Each component is designed to be:

1. **Single-responsibility**: Each component focuses on one aspect of the system.
2. **Loosely coupled**: Components interact through well-defined interfaces.
3. **Testable**: Each component can be tested in isolation.
4. **Configurable**: Behavior can be customized through configuration.
5. **Extensible**: New functionality can be added without modifying existing code.

## See Also

- [Examples](../examples.md) - Examples of using these components together 