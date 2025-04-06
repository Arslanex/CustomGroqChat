# Token Counter Tests Documentation

## Overview

This document provides detailed information about the unit tests implemented for the `token_counter` module. The tests ensure that the token counting functionality works correctly across various scenarios and handles edge cases appropriately.

## Test Coverage

The test suite covers the following aspects of the `token_counter` module:

1. **Message Token Counting**
   - Standard messages with content
   - Messages with function calls
   - Messages with empty content

2. **Multiple Messages Token Counting**
   - Token counting across multiple messages
   - Empty message lists
   - Overall formatting tokens

3. **Prompt Token Counting**
   - Standard text prompts
   - Empty prompts

4. **Request Token Counting**
   - Chat-based requests with messages
   - Completion-based requests with prompts
   - Unknown request formats

5. **Completion Token Estimation**
   - Requests with max_tokens specified
   - Requests without max_tokens
   - Requests with negative max_tokens

6. **Combined Token Counting**
   - Counting both prompt and completion tokens
   - Total token calculation

## Test Details

### Message Token Counting Tests

#### `test_count_tokens_in_message`
- **Purpose**: Verifies that the `count_tokens_in_message` function correctly counts tokens in a standard message.
- **Implementation**: Uses a mock encoder that counts words as tokens, and adds format tokens.
- **Validation**: Confirms that the token count matches the expected sum of content tokens plus format tokens.

#### `test_count_tokens_in_message_with_function_call`
- **Purpose**: Tests token counting in a message that contains a function call.
- **Implementation**: Creates a message with a function call and uses a mock encoder.
- **Validation**: Verifies that the token count includes both format tokens and function call tokens.

#### `test_count_tokens_in_message_empty_content`
- **Purpose**: Tests token counting in a message with empty content.
- **Implementation**: Creates a message with empty content and uses a mock encoder.
- **Validation**: Confirms that only format tokens are counted when content is empty.

### Multiple Messages Token Counting Tests

#### `test_count_tokens_in_messages_empty_list`
- **Purpose**: Tests token counting for an empty list of messages.
- **Implementation**: Passes an empty list to the `count_tokens_in_messages` function.
- **Validation**: Verifies that zero tokens are counted for an empty list.

#### `test_count_tokens_in_messages`
- **Purpose**: Tests token counting across multiple messages with overall formatting.
- **Implementation**: Mocks the `count_tokens_in_message` function to return specific values for each message.
- **Validation**: Confirms that the total includes tokens from all messages plus overall formatting tokens.

### Prompt Token Counting Tests

#### `test_count_tokens_in_prompt`
- **Purpose**: Tests token counting in a standard text prompt.
- **Implementation**: Uses a mock encoder to count words as tokens.
- **Validation**: Verifies that the correct number of tokens is counted.

#### `test_count_tokens_in_prompt_empty`
- **Purpose**: Tests token counting in an empty prompt.
- **Implementation**: Passes an empty string to the `count_tokens_in_prompt` function.
- **Validation**: Confirms that zero tokens are counted for an empty prompt.

### Request Token Counting Tests

#### `test_count_tokens_in_request_with_messages`
- **Purpose**: Tests token counting in a chat-based request with messages.
- **Implementation**: Mocks the `count_tokens_in_messages` function to return a specific value.
- **Validation**: Verifies that the token count from messages is used for the request.

#### `test_count_tokens_in_request_with_prompt`
- **Purpose**: Tests token counting in a completion-based request with a prompt.
- **Implementation**: Mocks the `count_tokens_in_prompt` function to return a specific value.
- **Validation**: Confirms that the token count from the prompt is used for the request.

#### `test_count_tokens_in_request_unknown_format`
- **Purpose**: Tests token counting when the request format is unknown.
- **Implementation**: Creates a request that doesn't contain messages or a prompt.
- **Validation**: Verifies that the default token count is used for unknown formats.

### Completion Token Estimation Tests

#### `test_estimate_completion_tokens_with_max_tokens`
- **Purpose**: Tests completion token estimation when max_tokens is specified.
- **Implementation**: Creates a request with a specified max_tokens value.
- **Validation**: Confirms that the estimated completion tokens match the specified max_tokens.

#### `test_estimate_completion_tokens_without_max_tokens`
- **Purpose**: Tests completion token estimation when max_tokens is not specified.
- **Implementation**: Creates a request without a max_tokens field.
- **Validation**: Verifies that the default token count is used when max_tokens is missing.

#### `test_estimate_completion_tokens_with_negative_max_tokens`
- **Purpose**: Tests completion token estimation when max_tokens is negative.
- **Implementation**: Creates a request with a negative max_tokens value.
- **Validation**: Confirms that the default token count is used when max_tokens is invalid.

### Combined Token Counting Tests

#### `test_count_request_and_completion_tokens`
- **Purpose**: Tests counting both request and completion tokens.
- **Implementation**: Mocks both `count_tokens_in_request` and `estimate_completion_tokens` functions.
- **Validation**: Verifies that the result includes prompt tokens, completion tokens, and the correct total.

## Implementation Notes

### Mock Encoding

The tests use a custom `MockEncoding` class to simulate the behavior of token encoders:

```python
class MockEncoding:
    def encode(self, text):
        # Simple mock that returns 1 token per word
        if not text:
            return []
        return text.split()
```

This mock provides a deterministic way to count tokens that's easy to reason about in tests.

### Test Fixtures

The tests create several fixtures to be reused across multiple tests:

- Sample messages with different roles
- A message with a function call
- A list of messages for conversation tests
- Sample request data for different types of requests

### Mocking Strategies

The tests use several mocking strategies:

1. **Function Patching**: Uses `unittest.mock.patch` to replace functions with mock versions.
2. **Return Values**: Sets specific return values for mocked functions.
3. **Side Effects**: Sets up mock functions to return different values on each call.
4. **Call Verification**: Validates that functions are called with the expected arguments.

## Running the Tests

The tests can be run using Python's unittest module:

```bash
python -m unittest test.token_counter_tests
```

When all tests pass, you should see output similar to:

```
..............
----------------------------------------------------------------------
Ran 14 tests in 0.002s

OK
```

## Integration with the Token Counter

These tests help ensure the reliability of the token counter, which is a critical component for:

1. Managing API rate limits based on token usage
2. Predicting token usage before making API calls
3. Tracking and monitoring token consumption
4. Optimizing prompts and requests for efficiency 