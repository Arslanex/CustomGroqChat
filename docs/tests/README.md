# CustomGroqChat Testing Documentation

This directory contains documentation related to testing the CustomGroqChat library, explaining test suites, methodologies, and coverage.

## Overview

Testing is a critical part of ensuring the reliability and correctness of the CustomGroqChat library. The tests cover various aspects of the library's functionality, from individual components to integration tests.

## Test Documentation

### Component Tests

These documents detail the testing approaches for each core component:

- [API Client Tests](api_client_tests.md) - Tests for API communication functions
- [Config Loader Tests](config_loader_tests.md) - Tests for configuration management
- [Queue Manager Tests](queue_manager_testing.md) - Tests for request queuing and prioritization
- [Rate Limit Handler Tests](rate_limit_handler_tests.md) - Tests for rate limiting logic
- [Token Counter Tests](token_counter_tests.md) - Tests for token counting accuracy

### Test Coverage Areas

The tests cover several critical areas:

1. **Functionality Testing** - Verifies that each component performs its intended functions correctly
2. **Error Handling** - Ensures proper handling of error cases and edge conditions
3. **Integration** - Tests components working together as a system
4. **Performance** - Measures response times and throughput under various conditions
5. **Rate Limiting** - Verifies that rate limiting works correctly under load

## Running Tests

To run the tests for CustomGroqChat:

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_api_client.py

# Run with coverage report
pytest --cov=CustomGroqChat
```

## Test Requirements

The test suite requires:

- pytest
- pytest-cov (for coverage reporting)
- pytest-asyncio (for testing async code)
- responses (for mocking HTTP responses)

Install these dependencies with:

```bash
pip install pytest pytest-cov pytest-asyncio responses
```

## Contributing Tests

When contributing new tests, please follow these guidelines:

1. Write tests for both success cases and failure cases
2. Mock external dependencies (especially the GROQ API)
3. Use async testing patterns for async functions
4. Ensure tests are deterministic and don't depend on external state
5. Maintain high test coverage for all new code

## See Also

- [Scripts Documentation](../scripts/) - Documentation for the components being tested
- [Examples](../examples.md) - Examples that demonstrate functionality verified by these tests 