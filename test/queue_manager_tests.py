import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from CutomGroqChat.queue_manager import QueueManager
from CutomGroqChat.api_client import APIClient
from CutomGroqChat.rate_limit_handler import RateLimitHandler


class TestQueueManager(unittest.TestCase):
    """Test cases for the QueueManager class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.api_client_mock = AsyncMock(spec=APIClient)
        self.rate_limit_handler_mock = MagicMock(spec=RateLimitHandler)
        self.queue_manager = QueueManager(self.api_client_mock, self.rate_limit_handler_mock)

    def tearDown(self):
        """Clean up after each test method."""
        # Stop the queue manager if it's running
        self.queue_manager.stop()

    def test_init(self):
        """Test QueueManager initialization."""
        self.assertEqual(self.queue_manager.api_client, self.api_client_mock)
        self.assertEqual(self.queue_manager.rate_limit_handler, self.rate_limit_handler_mock)
        self.assertEqual(self.queue_manager.high_priority_queue, [])
        self.assertEqual(self.queue_manager.normal_priority_queue, [])
        self.assertEqual(self.queue_manager.low_priority_queue, [])
        self.assertFalse(self.queue_manager.running)
        self.assertIsNone(self.queue_manager.processing_task)
        self.assertEqual(self.queue_manager.request_map, {})

    def test_start_stop(self):
        """Test starting and stopping the queue manager."""
        # Initially not running
        self.assertFalse(self.queue_manager.running)
        
        # Start the queue manager
        self.queue_manager.start()
        self.assertTrue(self.queue_manager.running)
        
        # Stop the queue manager
        self.queue_manager.stop()
        self.assertFalse(self.queue_manager.running)
        self.assertIsNone(self.queue_manager.processing_task)

    def test_get_queue_length(self):
        """Test getting queue lengths."""
        # Initial state - all queues empty
        queue_lengths = self.queue_manager.get_queue_length()
        self.assertEqual(queue_lengths["high"], 0)
        self.assertEqual(queue_lengths["normal"], 0)
        self.assertEqual(queue_lengths["low"], 0)
        self.assertEqual(queue_lengths["total"], 0)
        
        # Add items to queues (directly for testing)
        self.queue_manager.high_priority_queue.append({"id": "high1"})
        self.queue_manager.normal_priority_queue.append({"id": "normal1"})
        self.queue_manager.normal_priority_queue.append({"id": "normal2"})
        self.queue_manager.low_priority_queue.append({"id": "low1"})
        
        # Check updated lengths
        queue_lengths = self.queue_manager.get_queue_length()
        self.assertEqual(queue_lengths["high"], 1)
        self.assertEqual(queue_lengths["normal"], 2)
        self.assertEqual(queue_lengths["low"], 1)
        self.assertEqual(queue_lengths["total"], 4)

    def test_get_next_request(self):
        """Test getting next request based on priority."""
        # Add items to all priority queues
        high_request = {"id": "high1", "priority": "high"}
        normal_request = {"id": "normal1", "priority": "normal"}
        low_request = {"id": "low1", "priority": "low"}
        
        self.queue_manager.high_priority_queue.append(high_request)
        self.queue_manager.normal_priority_queue.append(normal_request)
        self.queue_manager.low_priority_queue.append(low_request)
        
        # Should get high priority first
        next_request = self.queue_manager.get_next_request()
        self.assertEqual(next_request, high_request)
        self.assertEqual(len(self.queue_manager.high_priority_queue), 0)
        
        # Then normal priority
        next_request = self.queue_manager.get_next_request()
        self.assertEqual(next_request, normal_request)
        self.assertEqual(len(self.queue_manager.normal_priority_queue), 0)
        
        # Then low priority
        next_request = self.queue_manager.get_next_request()
        self.assertEqual(next_request, low_request)
        self.assertEqual(len(self.queue_manager.low_priority_queue), 0)
        
        # Then none
        next_request = self.queue_manager.get_next_request()
        self.assertIsNone(next_request)

    def test_enqueue_request(self):
        """Test enqueueing requests with different priorities."""
        # Create an event loop for testing async functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Patch ensure_processing to prevent starting the process loop
            with patch.object(QueueManager, 'ensure_processing', AsyncMock()):
                # Enqueue with different priorities
                high_id = loop.run_until_complete(self.queue_manager.enqueue_request(
                    endpoint="/test", payload={"test": "high"}, token_count=10, priority="high"
                ))
                normal_id = loop.run_until_complete(self.queue_manager.enqueue_request(
                    endpoint="/test", payload={"test": "normal"}, token_count=10, priority="normal"
                ))
                low_id = loop.run_until_complete(self.queue_manager.enqueue_request(
                    endpoint="/test", payload={"test": "low"}, token_count=10, priority="low"
                ))
                
                # Verify requests were added to appropriate queues
                self.assertEqual(len(self.queue_manager.high_priority_queue), 1)
                self.assertEqual(len(self.queue_manager.normal_priority_queue), 1)
                self.assertEqual(len(self.queue_manager.low_priority_queue), 1)
                
                # Verify request map contains all requests
                self.assertEqual(len(self.queue_manager.request_map), 3)
                self.assertIn(high_id, self.queue_manager.request_map)
                self.assertIn(normal_id, self.queue_manager.request_map)
                self.assertIn(low_id, self.queue_manager.request_map)
                
                # Verify ensure_processing was called
                self.queue_manager.ensure_processing.assert_called()
        finally:
            loop.close()

    def test_enqueue_request_invalid_priority(self):
        """Test enqueueing a request with an invalid priority."""
        # Create an event loop for testing async functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            with patch.object(QueueManager, 'ensure_processing', AsyncMock()):
                with self.assertRaises(ValueError):
                    loop.run_until_complete(self.queue_manager.enqueue_request(
                        endpoint="/test", payload={"test": "invalid"}, 
                        token_count=10, priority="invalid"
                    ))
        finally:
            loop.close()

    def test_cancel_request(self):
        """Test cancelling a request."""
        # Create an event loop for testing async functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Patch ensure_processing
            with patch.object(QueueManager, 'ensure_processing', AsyncMock()):
                # Enqueue requests
                high_id = loop.run_until_complete(self.queue_manager.enqueue_request(
                    endpoint="/test", payload={"test": "high"}, token_count=10, priority="high"
                ))
                normal_id = loop.run_until_complete(self.queue_manager.enqueue_request(
                    endpoint="/test", payload={"test": "normal"}, token_count=10, priority="normal"
                ))
                
                # Cancel high priority request
                result = loop.run_until_complete(self.queue_manager.cancel_request(high_id))
                self.assertTrue(result)
                self.assertEqual(len(self.queue_manager.high_priority_queue), 0)
                self.assertNotIn(high_id, self.queue_manager.request_map)
                
                # Try to cancel non-existent request
                result = loop.run_until_complete(self.queue_manager.cancel_request("non-existent"))
                self.assertFalse(result)
                
                # Cancel normal priority request
                result = loop.run_until_complete(self.queue_manager.cancel_request(normal_id))
                self.assertTrue(result)
                self.assertEqual(len(self.queue_manager.normal_priority_queue), 0)
                self.assertNotIn(normal_id, self.queue_manager.request_map)
        finally:
            loop.close()

    def test_process_request(self):
        """Test processing a single request."""
        # Create an event loop for testing async functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Setup mock response
            self.api_client_mock.post_request.return_value = {"response": "test"}
            
            # Create mock callback
            callback_mock = AsyncMock()
            
            # Create test request
            request = {
                "id": "test-id",
                "endpoint": "/test",
                "payload": {"test": "data"},
                "token_count": 10,
                "callback": callback_mock,
                "priority": "high"
            }
            
            # Process the request
            loop.run_until_complete(self.queue_manager._process_request(request))
            
            # Verify API client was called
            self.api_client_mock.post_request.assert_called_once_with("/test", {"test": "data"})
            
            # Verify rate limit handler was updated
            self.rate_limit_handler_mock.update_counters.assert_called_once_with(10)
            
            # Verify callback was called with the response
            callback_mock.assert_called_once_with({"response": "test"})
        finally:
            loop.close()

    def test_process_request_with_error(self):
        """Test processing a request that results in an error."""
        # Create an event loop for testing async functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Setup mock to raise an exception
            self.api_client_mock.post_request.side_effect = Exception("Test error")
            
            # Create mock callback
            callback_mock = AsyncMock()
            
            # Create test request
            request = {
                "id": "test-id",
                "endpoint": "/test",
                "payload": {"test": "data"},
                "token_count": 10,
                "callback": callback_mock,
                "priority": "high"
            }
            
            # Process the request
            loop.run_until_complete(self.queue_manager._process_request(request))
            
            # Verify callback was called with the error
            callback_mock.assert_called_once()
            args = callback_mock.call_args[0][0]
            self.assertIn("error", args)
            self.assertEqual(args["error"], "Test error")
        finally:
            loop.close()

    def test_ensure_processing(self):
        """Test ensuring the processing task is running."""
        # Create an event loop for testing async functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Test with running=False
            self.queue_manager.running = False
            loop.run_until_complete(self.queue_manager.ensure_processing())
            self.assertIsNone(self.queue_manager.processing_task)
            
            # Test with running=True
            with patch.object(asyncio, 'create_task'):
                self.queue_manager.running = True
                loop.run_until_complete(self.queue_manager.ensure_processing())
                asyncio.create_task.assert_called_once()
        finally:
            loop.close()

    def test_process_queue(self):
        """Test the queue processing loop."""
        # Create an event loop for testing async functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Setup rate limit handler mock
            self.rate_limit_handler_mock.can_make_request.return_value = (True, [])
            
            # Setup API client mock
            self.api_client_mock.post_request.return_value = {"response": "test"}
            
            # Setup callback mock
            callback_mock = AsyncMock()
            
            # Add a request to the high priority queue
            request = {
                "id": "test-id",
                "endpoint": "/test",
                "payload": {"test": "data"},
                "token_count": 10,
                "callback": callback_mock,
                "priority": "high",
                "enqueue_time": loop.time()
            }
            
            self.queue_manager.high_priority_queue.append(request)
            self.queue_manager.request_map["test-id"] = request
            
            # Make _process_queue exit after one iteration
            self.queue_manager.running = True
            
            # Patch sleep to prevent infinite loop and exit after processing
            with patch.object(asyncio, 'sleep', AsyncMock()) as sleep_mock:
                # Setup side effect to stop the loop after first iteration
                def sleep_side_effect(*args, **kwargs):
                    self.queue_manager.running = False
                    future = asyncio.Future()
                    future.set_result(None)
                    return future
                
                sleep_mock.side_effect = sleep_side_effect
                
                # Run the process queue method
                loop.run_until_complete(self.queue_manager._process_queue())
                
                # Verify the request was processed
                callback_mock.assert_called_once()
                self.assertEqual(len(self.queue_manager.high_priority_queue), 0)
                self.assertEqual(len(self.queue_manager.request_map), 0)
        finally:
            loop.close()

    def test_get_queue_status(self):
        """Test getting the queue status."""
        # Setup rate limit handler mock
        self.rate_limit_handler_mock.get_status.return_value = {"rate_limit": "status"}
        
        # Add items to queues
        self.queue_manager.high_priority_queue.append({"id": "high1"})
        self.queue_manager.normal_priority_queue.append({"id": "normal1"})
        
        # Get status
        status = self.queue_manager.get_queue_status()
        
        # Verify status
        self.assertEqual(status["queue_lengths"]["high"], 1)
        self.assertEqual(status["queue_lengths"]["normal"], 1)
        self.assertEqual(status["queue_lengths"]["low"], 0)
        self.assertEqual(status["queue_lengths"]["total"], 2)
        self.assertEqual(status["total_queue_length"], 2)
        self.assertEqual(status["running"], False)
        self.assertEqual(status["rate_limits"], {"rate_limit": "status"})


if __name__ == "__main__":
    unittest.main() 