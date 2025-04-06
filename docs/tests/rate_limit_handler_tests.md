# RateLimitHandler Tests Documentation

## Overview

This document provides detailed information about the unit tests implemented for the `RateLimitHandler` class. The tests ensure that the rate limiting functionality works correctly, handles errors appropriately, and manages request and token limits as expected.

## Test Coverage

The test suite covers the following aspects of the `RateLimitHandler` class:

1. **Initialization**
   - Default configuration
   - Custom configuration
   - Unlimited configuration

2. **Rate Limit Checking**
   - Valid requests
   - Invalid inputs
   - Minute-based request limit validation
   - Minute-based token limit validation
   - Day-based request limit validation
   - Day-based token limit validation
   - Multiple limit validations
   - Unlimited limits

3. **Counter Management**
   - Minute counter reset
   - Day counter reset
   - Counter updates

4. **Request Validation**
   - Valid request checking
   - Invalid input handling
   - Exception raising for various limit types

5. **Status Reporting**
   - Status information retrieval
   - Formatting of unlimited limits

6. **Integration**
   - Full lifecycle of request handling
   - Time-based counter resets

## Test Details

### Initialization Tests

#### `test_initialization`
- **Purpose**: Verifies that the `RateLimitHandler` initializes correctly with different configurations.
- **Validation**: 
  - Default configuration values are set correctly
  - Custom configuration values are applied as expected
  - Unlimited configuration (-1 values) is handled correctly

### Rate Limit Checking Tests

#### `test_can_make_request_valid`
- **Purpose**: Tests the `can_make_request` method with valid parameters.
- **Validation**: Confirms that a request can be made when limits are not exceeded.

#### `test_can_make_request_invalid_input`
- **Purpose**: Tests the `can_make_request` method with invalid input.
- **Validation**: Ensures that a `TypeError` is raised when a non-integer token count is provided.

#### `test_can_make_request_minute_request_limit_exceeded`
- **Purpose**: Tests that the minute request limit is properly enforced.
- **Validation**: Verifies that the request is rejected when the minute request limit is exceeded.

#### `test_can_make_request_minute_token_limit_exceeded`
- **Purpose**: Tests that the minute token limit is properly enforced.
- **Validation**: Verifies that the request is rejected when the minute token limit would be exceeded.

#### `test_can_make_request_day_request_limit_exceeded`
- **Purpose**: Tests that the day request limit is properly enforced.
- **Validation**: Verifies that the request is rejected when the day request limit is exceeded.

#### `test_can_make_request_day_token_limit_exceeded`
- **Purpose**: Tests that the day token limit is properly enforced.
- **Validation**: Verifies that the request is rejected when the day token limit would be exceeded.

#### `test_can_make_request_multiple_limits_exceeded`
- **Purpose**: Tests the handling of multiple exceeded limits.
- **Validation**: Confirms that all applicable limit reasons are returned when multiple limits are exceeded.

#### `test_can_make_request_unlimited`
- **Purpose**: Tests the behavior with unlimited limits.
- **Validation**: Ensures that requests can be made regardless of counter values when limits are set to unlimited (-1).

### Counter Management Tests

#### `test_reset_minute_counters`
- **Purpose**: Tests the resetting of minute-based counters.
- **Implementation**: Uses `unittest.mock.patch` to control time.
- **Validation**: 
  - Counters remain unchanged when less than a minute has passed
  - Counters are reset when more than a minute has passed

#### `test_reset_day_counters`
- **Purpose**: Tests the resetting of day-based counters.
- **Implementation**: Uses `unittest.mock.patch` to control time.
- **Validation**: 
  - Counters remain unchanged when less than a day has passed
  - Counters are reset when more than a day has passed

#### `test_update_counters`
- **Purpose**: Tests the updating of counters after requests.
- **Validation**: Confirms that all counters are correctly incremented.

### Request Validation Tests

#### `test_check_request_valid`
- **Purpose**: Tests the `check_request` method with valid parameters.
- **Validation**: Ensures that no exception is raised for valid requests.

#### `test_check_request_invalid_input`
- **Purpose**: Tests the `check_request` method with invalid input.
- **Validation**: Confirms that a `TypeError` is raised for non-integer token counts.

#### `test_check_request_minute_request_limit_exceeded`
- **Purpose**: Tests exception raising when minute request limit is exceeded.
- **Validation**: Verifies that a `RateLimitExceededException` with the correct parameters is raised.

#### `test_check_request_minute_token_limit_exceeded`
- **Purpose**: Tests exception raising when minute token limit is exceeded.
- **Validation**: Verifies that a `RateLimitExceededException` with the correct parameters is raised.

#### `test_check_request_day_request_limit_exceeded`
- **Purpose**: Tests exception raising when day request limit is exceeded.
- **Validation**: Verifies that a `RateLimitExceededException` with the correct parameters is raised.

#### `test_check_request_day_token_limit_exceeded`
- **Purpose**: Tests exception raising when day token limit is exceeded.
- **Validation**: Verifies that a `RateLimitExceededException` with the correct parameters is raised.

### Status Reporting Tests

#### `test_get_status`
- **Purpose**: Tests the retrieval of rate limit status information.
- **Validation**: Confirms that the status dictionary contains accurate information about current usage and limits.

#### `test_get_status_unlimited`
- **Purpose**: Tests status reporting with unlimited limits.
- **Validation**: Verifies that unlimited limits are correctly formatted in the status report.

### Integration Tests

#### `test_integration_scenario`
- **Purpose**: Tests a full request lifecycle with time-based resets.
- **Implementation**: Uses `unittest.mock.patch` to control time progression.
- **Validation**: 
  - Requests succeed until limits are reached
  - Minute-based limits are reset after a minute
  - Day-based limits are reset after a day
  - Status reporting reflects the current state

## Implementation Notes

### Test Setup
The test suite uses the `unittest` framework with the following setup:
- A common default configuration
- An instance of `RateLimitHandler` for most tests
- Specific configurations for specialized tests

### Mock Patching
Several tests use `unittest.mock.patch` to:
- Control time progression for testing counter resets
- Simulate the passage of time in integration scenarios

### Exception Testing
The tests validate exception handling in various scenarios:
- Type checking for inputs
- Rate limit exceeded conditions
- Exception attribute correctness

## Running the Tests

The tests can be run using Python's unittest module:

```bash
python -m unittest test.rate_limit_handler_tests
```

When all tests pass, you should see output similar to:

```
.......................
----------------------------------------------------------------------
Ran 23 tests in 0.005s

OK
```

## Integration with GROQ API Client

The `RateLimitHandler` is a critical component for managing API rate limits for the GROQ API client. These tests ensure that the rate limiting functionality works correctly to prevent quota overages and properly handle rate limit errors.

The handler works with the following types of limits:
- Request limits per minute
- Request limits per day
- Token limits per minute
- Token limits per day

Each limit can be set to a positive integer for a specific limit or -1 for unlimited. 