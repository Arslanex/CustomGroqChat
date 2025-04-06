# QueueManager Tests Documentation

## Overview

This document provides detailed information about the unit tests implemented for the `QueueManager` class. The tests ensure that the priority-based asynchronous request queue works correctly, respects rate limits, and properly handles request callbacks and cancellations.

## Test Coverage

The test suite covers the following aspects of the `QueueManager` class:

1. **Initialization**
   - Default configuration
   - APIClient and RateLimitHandler dependency injection

2. **Queue Management**
   - Starting and stopping the queue manager
   - Queue status reporting
   - Queue length monitoring

3. **Request Processing**
   - Basic request enqueuing and execution
   - Priority-based request ordering
   - Rate limit respecting
   - Callback invocation
   - Error handling in request processing

4. **Priority Handling**
   - High priority queue processing
   - Normal priority queue processing
   - Low priority queue processing
   - Strict priority ordering enforcement

5. **Request Cancellation**
   - Valid request cancellation
   - Invalid request ID handling
   - Queue state after cancellation

6. **Rate Limit Integration**
   - Request requeuing when rate limited
   - Continuing processing after rate limits allow
   - Rate limit status reporting

7. **Async Processing**
   - Event loop integration
   - Asynchronous callback handling
   - Task cancellation and cleanup

## Test Details

### Initialization Tests

#### `test_initialization`
- **Purpose**: Verifies that the `QueueManager` initializes correctly with the provided dependencies.
- **Validation**: 
  - Dependencies are properly stored
  - Queues are initially empty
  - Running flag is set to False
  - Request map is empty

### Queue Management Tests

#### `test_start_stop`
- **Purpose**: Tests the starting and stopping of the queue manager.
- **Validation**: 
  - Running flag is True after start
  - Running flag is False after stop
  - Processing task is None after stop

#### `test_get_queue_status`
- **Purpose**: Tests the retrieval of queue status information.
- **Validation**: 
  - Queue lengths are correctly reported
  - Running state is accurately reflected
  - Rate limit status is included

### Request Processing Tests

#### `test_basic_request_processing`
- **Purpose**: Tests the basic flow of enqueuing and processing a request.
- **Validation**: 
  - API client is called with correct parameters
  - Callback is invoked with the API response
  - Request is removed from the request map after processing

#### `test_process_request_with_error`
- **Purpose**: Tests handling of errors during request processing.
- **Validation**: 
  - Errors are caught and don't crash the queue processor
  - Callback is called with error information
  - Processing continues to next request

### Priority Handling Tests

#### `test_priority_order`
- **Purpose**: Tests that requests are processed in strict priority order.
- **Validation**: 
  - High priority requests are processed before normal priority
  - Normal priority requests are processed before low priority
  - Requests of same priority are processed in FIFO order

#### `test_high_priority_preemption`
- **Purpose**: Tests that high priority requests preempt lower priority ones.
- **Validation**: 
  - When a high priority request is added, it's processed next even if other requests are waiting

### Request Cancellation Tests

#### `test_request_cancellation`
- **Purpose**: Tests cancellation of a pending request.
- **Validation**: 
  - Request is removed from the appropriate queue
  - Request is removed from the request map
  - Cancel method returns True for successful cancellation

#### `test_invalid_cancellation`
- **Purpose**: Tests cancellation with an invalid request ID.
- **Validation**: 
  - Cancel method returns False for non-existent request IDs
  - No changes to the queues occur

### Rate Limit Integration Tests

#### `test_rate_limit_handling`
- **Purpose**: Tests integration with the rate limit handler.
- **Validation**: 
  - Requests are requeued when rate limits are hit
  - Processing continues when rate limits allow
  - Rate limit handler is consulted before processing requests

#### `test_rate_limit_counter_update`
- **Purpose**: Tests that rate limit counters are updated after successful processing.
- **Validation**: 
  - Rate limit handler's update_counters method is called with correct token count

### Async Processing Tests

#### `test_ensure_processing`
- **Purpose**: Tests that the processing loop is created when needed.
- **Validation**: 
  - Processing task is created when running and not already processing
  - Processing task is not recreated when already running

#### `test_processing_task_cancel`
- **Purpose**: Tests that the processing task is properly cancelled on stop.
- **Validation**: 
  - Task is cancelled when stop is called
  - Task reference is cleared after cancellation

## Implementation Notes

### Test Setup
The test suite uses the `pytest` framework with the following setup:
- Mock objects for `APIClient` and `RateLimitHandler`
- Fixture for creating a QueueManager instance
- Async test functions using pytest-asyncio

### Mock Usage
The tests use mock objects to:
- Simulate API client responses
- Control rate limit handler behavior
- Track callback execution
- Verify method calls and parameters

### Async Testing
Several tests use `asyncio` features to:
- Wait for asynchronous processing to complete
- Create and test callbacks
- Simulate the event loop for task creation and cancellation

### Fixtures
The tests use pytest fixtures for:
- Creating mock dependencies
- Setting up the QueueManager
- Ensuring cleanup after tests

## Running the Tests

The tests can be run using pytest:

```bash
pytest test/queue_manager_tests.py -v
```

When all tests pass, you should see output similar to:

```
test/queue_manager_tests.py::test_initialization PASSED
test/queue_manager_tests.py::test_start_stop PASSED
test/queue_manager_tests.py::test_get_queue_status PASSED
test/queue_manager_tests.py::test_basic_request_processing PASSED
test/queue_manager_tests.py::test_process_request_with_error PASSED
test/queue_manager_tests.py::test_priority_order PASSED
test/queue_manager_tests.py::test_high_priority_preemption PASSED
test/queue_manager_tests.py::test_request_cancellation PASSED
test/queue_manager_tests.py::test_invalid_cancellation PASSED
test/queue_manager_tests.py::test_rate_limit_handling PASSED
test/queue_manager_tests.py::test_rate_limit_counter_update PASSED
test/queue_manager_tests.py::test_ensure_processing PASSED
test/queue_manager_tests.py::test_processing_task_cancel PASSED

================= 13 passed in 0.47s =================
```

## Integration with GROQ API Client

The `QueueManager` is a critical component for managing the request queue and rate limiting for the GROQ API client. These tests ensure that the queue management functionality works correctly to:

- Enforce priority-based processing
- Respect API rate limits
- Handle callbacks and errors properly
- Allow request cancellation

The manager works with the following priority levels:
- High: For urgent requests that need immediate processing
- Normal: For standard priority requests
- Low: For non-urgent, background requests (default)

Each request is processed according to its priority level, and rate limits are checked before processing to ensure API quotas are not exceeded 