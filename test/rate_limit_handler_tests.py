"""
Unit tests for the rate_limit_handler module.

Tests:
- test_initialization
- test_can_make_request_valid
- test_can_make_request_invalid_input
- test_can_make_request_minute_request_limit_exceeded
- test_can_make_request_minute_token_limit_exceeded
- test_can_make_request_day_request_limit_exceeded
- test_can_make_request_day_token_limit_exceeded
- test_can_make_request_multiple_limits_exceeded
- test_can_make_request_unlimited
- test_reset_minute_counters
- test_reset_day_counters
- test_update_counters
- test_check_request_valid
- test_check_request_invalid_input
- test_check_request_minute_request_limit_exceeded
- test_check_request_minute_token_limit_exceeded
- test_check_request_day_request_limit_exceeded
- test_check_request_day_token_limit_exceeded
- test_get_status
- test_get_status_unlimited
- test_integration_scenario
"""

import unittest
import time
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List, Tuple

from CutomGroqChat.rate_limit_handler import RateLimitHandler
from CutomGroqChat.exceptions import RateLimitExceededException


class TestRateLimitHandler(unittest.TestCase):
    """Unit tests for the RateLimitHandler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.default_config = {
            "req_per_min": 60,
            "req_per_day": 1000,
            "tokens_per_min": 6000,
            "tokens_per_day": 250000
        }
        self.handler = RateLimitHandler(self.default_config)

    def test_initialization(self):
        """Test initialization with different configurations."""
        # Test with default config
        self.assertEqual(self.handler.req_per_min, 60)
        self.assertEqual(self.handler.req_per_day, 1000)
        self.assertEqual(self.handler.tokens_per_min, 6000)
        self.assertEqual(self.handler.tokens_per_day, 250000)
        self.assertEqual(self.handler.req_minute_counter, 0)
        self.assertEqual(self.handler.req_day_counter, 0)
        self.assertEqual(self.handler.tokens_minute_counter, 0)
        self.assertEqual(self.handler.tokens_day_counter, 0)

        # Test with custom config
        custom_config = {
            "req_per_min": 30,
            "req_per_day": 500,
            "tokens_per_min": 3000,
            "tokens_per_day": 120000
        }
        handler = RateLimitHandler(custom_config)
        self.assertEqual(handler.req_per_min, 30)
        self.assertEqual(handler.req_per_day, 500)
        self.assertEqual(handler.tokens_per_min, 3000)
        self.assertEqual(handler.tokens_per_day, 120000)

        # Test with unlimited config (-1)
        unlimited_config = {
            "req_per_min": -1,
            "req_per_day": -1,
            "tokens_per_min": -1,
            "tokens_per_day": -1
        }
        handler = RateLimitHandler(unlimited_config)
        self.assertEqual(handler.req_per_min, -1)
        self.assertEqual(handler.req_per_day, -1)
        self.assertEqual(handler.tokens_per_min, -1)
        self.assertEqual(handler.tokens_per_day, -1)

    def test_can_make_request_valid(self):
        """Test can_make_request with valid parameters."""
        # Test with valid token count
        can_make, reasons = self.handler.can_make_request(100)
        self.assertTrue(can_make)
        self.assertEqual(len(reasons), 0)

    def test_can_make_request_invalid_input(self):
        """Test can_make_request with invalid input."""
        # Test with non-integer token count
        with self.assertRaises(TypeError):
            self.handler.can_make_request("100")

    def test_can_make_request_minute_request_limit_exceeded(self):
        """Test can_make_request when minute request limit is exceeded."""
        # Set up the handler to exceed the minute request limit
        self.handler.req_minute_counter = self.handler.req_per_min
        can_make, reasons = self.handler.can_make_request(100)
        self.assertFalse(can_make)
        self.assertIn("Rate limit exceeded - Minute Request Limit", reasons)

    def test_can_make_request_minute_token_limit_exceeded(self):
        """Test can_make_request when minute token limit is exceeded."""
        # Set up the handler to exceed the minute token limit
        self.handler.tokens_minute_counter = self.handler.tokens_per_min - 50
        can_make, reasons = self.handler.can_make_request(100)
        self.assertFalse(can_make)
        self.assertIn("Rate limit exceeded - Minute Token Limit", reasons)

    def test_can_make_request_day_request_limit_exceeded(self):
        """Test can_make_request when day request limit is exceeded."""
        # Set up the handler to exceed the day request limit
        self.handler.req_day_counter = self.handler.req_per_day
        can_make, reasons = self.handler.can_make_request(100)
        self.assertFalse(can_make)
        self.assertIn("Rate limit exceeded - Daily Request Limit", reasons)

    def test_can_make_request_day_token_limit_exceeded(self):
        """Test can_make_request when day token limit is exceeded."""
        # Set up the handler to exceed the day token limit
        self.handler.tokens_day_counter = self.handler.tokens_per_day - 50
        can_make, reasons = self.handler.can_make_request(100)
        self.assertFalse(can_make)
        self.assertIn("Rate limit exceeded - Daily Token Limit", reasons)

    def test_can_make_request_multiple_limits_exceeded(self):
        """Test can_make_request when multiple limits are exceeded."""
        # Set up the handler to exceed multiple limits
        self.handler.req_minute_counter = self.handler.req_per_min
        self.handler.tokens_day_counter = self.handler.tokens_per_day - 50
        can_make, reasons = self.handler.can_make_request(100)
        self.assertFalse(can_make)
        self.assertEqual(len(reasons), 2)
        self.assertIn("Rate limit exceeded - Minute Request Limit", reasons)
        self.assertIn("Rate limit exceeded - Daily Token Limit", reasons)

    def test_can_make_request_unlimited(self):
        """Test can_make_request with unlimited limits."""
        # Set up the handler with unlimited limits
        unlimited_config = {
            "req_per_min": -1,
            "req_per_day": -1,
            "tokens_per_min": -1,
            "tokens_per_day": -1
        }
        handler = RateLimitHandler(unlimited_config)

        # Set very high counter values
        handler.req_minute_counter = 10000
        handler.req_day_counter = 100000
        handler.tokens_minute_counter = 1000000
        handler.tokens_day_counter = 10000000

        # Test that request can still be made
        can_make, reasons = handler.can_make_request(1000000)
        self.assertTrue(can_make)
        self.assertEqual(len(reasons), 0)

    @patch('time.time')
    def test_reset_minute_counters(self, mock_time):
        """Test _reset_minute_counters method."""
        # Setup initial conditions
        self.handler.req_minute_counter = 30
        self.handler.tokens_minute_counter = 3000
        initial_time = 1000.0
        self.handler._last_minute_reset = initial_time

        # Test when less than a minute has passed
        mock_time.return_value = initial_time + 30  # 30 seconds later
        self.handler._reset_minute_counters()
        self.assertEqual(self.handler.req_minute_counter, 30)
        self.assertEqual(self.handler.tokens_minute_counter, 3000)

        # Test when more than a minute has passed
        mock_time.return_value = initial_time + 61  # 61 seconds later
        self.handler._reset_minute_counters()
        self.assertEqual(self.handler.req_minute_counter, 0)
        self.assertEqual(self.handler.tokens_minute_counter, 0)
        self.assertEqual(self.handler._last_minute_reset, initial_time + 61)

    @patch('time.time')
    def test_reset_day_counters(self, mock_time):
        """Test _reset_day_counters method."""
        # Setup initial conditions
        self.handler.req_day_counter = 500
        self.handler.tokens_day_counter = 50000
        initial_time = 1000.0
        self.handler._last_day_reset = initial_time

        # Test when less than a day has passed
        mock_time.return_value = initial_time + 3600  # 1 hour later
        self.handler._reset_day_counters()
        self.assertEqual(self.handler.req_day_counter, 500)
        self.assertEqual(self.handler.tokens_day_counter, 50000)

        # Test when more than a day has passed
        mock_time.return_value = initial_time + 86401  # 24 hours + 1 second later
        self.handler._reset_day_counters()
        self.assertEqual(self.handler.req_day_counter, 0)
        self.assertEqual(self.handler.tokens_day_counter, 0)
        self.assertEqual(self.handler._last_day_reset, initial_time + 86401)

    def test_update_counters(self):
        """Test update_counters method."""
        # Initial state
        self.assertEqual(self.handler.tokens_minute_counter, 0)
        self.assertEqual(self.handler.tokens_day_counter, 0)
        self.assertEqual(self.handler.req_minute_counter, 0)
        self.assertEqual(self.handler.req_day_counter, 0)

        # Update counters
        self.handler.update_counters(100)
        self.assertEqual(self.handler.tokens_minute_counter, 100)
        self.assertEqual(self.handler.tokens_day_counter, 100)
        self.assertEqual(self.handler.req_minute_counter, 1)
        self.assertEqual(self.handler.req_day_counter, 1)

        # Update again
        self.handler.update_counters(200)
        self.assertEqual(self.handler.tokens_minute_counter, 300)
        self.assertEqual(self.handler.tokens_day_counter, 300)
        self.assertEqual(self.handler.req_minute_counter, 2)
        self.assertEqual(self.handler.req_day_counter, 2)

    def test_check_request_valid(self):
        """Test check_request with valid parameters."""
        # Should not raise an exception
        self.handler.check_request(100)  # This should not raise an exception

    def test_check_request_invalid_input(self):
        """Test check_request with invalid input."""
        # Test with non-integer token count
        with self.assertRaises(TypeError):
            self.handler.check_request("100")

    def test_check_request_minute_request_limit_exceeded(self):
        """Test check_request when minute request limit is exceeded."""
        # Set up the handler to exceed the minute request limit
        self.handler.req_minute_counter = self.handler.req_per_min

        with self.assertRaises(RateLimitExceededException) as context:
            self.handler.check_request(100, strictly=True)

        exception = context.exception
        self.assertEqual(exception.limit_type, "request")
        self.assertEqual(exception.time_period, "minute")
        self.assertEqual(exception.current_value, self.handler.req_minute_counter)
        self.assertEqual(exception.limit_value, self.handler.req_per_min)

    def test_check_request_minute_token_limit_exceeded(self):
        """Test check_request when minute token limit is exceeded."""
        # Set up the handler to exceed the minute token limit
        self.handler.tokens_minute_counter = self.handler.tokens_per_min - 50

        with self.assertRaises(RateLimitExceededException) as context:
            self.handler.check_request(100, strictly=True)

        exception = context.exception
        self.assertEqual(exception.limit_type, "token")
        self.assertEqual(exception.time_period, "minute")
        self.assertEqual(exception.current_value, self.handler.tokens_minute_counter)
        self.assertEqual(exception.limit_value, self.handler.tokens_per_min)

    def test_check_request_day_request_limit_exceeded(self):
        """Test check_request when day request limit is exceeded."""
        # Set up the handler to exceed the day request limit
        self.handler.req_day_counter = self.handler.req_per_day

        with self.assertRaises(RateLimitExceededException) as context:
            self.handler.check_request(100, strictly=True)

        exception = context.exception
        self.assertEqual(exception.limit_type, "request")
        self.assertEqual(exception.time_period, "day")
        self.assertEqual(exception.current_value, self.handler.req_day_counter)
        self.assertEqual(exception.limit_value, self.handler.req_per_day)

    def test_check_request_day_token_limit_exceeded(self):
        """Test check_request when day token limit is exceeded."""
        # Set up the handler to exceed the day token limit
        self.handler.tokens_day_counter = self.handler.tokens_per_day - 50

        with self.assertRaises(RateLimitExceededException) as context:
            self.handler.check_request(100, strictly=True)

        exception = context.exception
        self.assertEqual(exception.limit_type, "token")
        self.assertEqual(exception.time_period, "day")
        self.assertEqual(exception.current_value, self.handler.tokens_day_counter)
        self.assertEqual(exception.limit_value, self.handler.tokens_per_day)

    def test_get_status(self):
        """Test get_status method."""
        # Setup initial state
        self.handler.req_minute_counter = 10
        self.handler.req_day_counter = 100
        self.handler.tokens_minute_counter = 1000
        self.handler.tokens_day_counter = 10000

        # Get status
        status = self.handler.get_status()

        # Verify status
        self.assertEqual(status["requests"]["minute"]["current"], 10)
        self.assertEqual(status["requests"]["minute"]["limit"], 60)
        self.assertEqual(status["requests"]["minute"]["display"], "10/60")

        self.assertEqual(status["requests"]["day"]["current"], 100)
        self.assertEqual(status["requests"]["day"]["limit"], 1000)
        self.assertEqual(status["requests"]["day"]["display"], "100/1000")

        self.assertEqual(status["tokens"]["minute"]["current"], 1000)
        self.assertEqual(status["tokens"]["minute"]["limit"], 6000)
        self.assertEqual(status["tokens"]["minute"]["display"], "1000/6000")

        self.assertEqual(status["tokens"]["day"]["current"], 10000)
        self.assertEqual(status["tokens"]["day"]["limit"], 250000)
        self.assertEqual(status["tokens"]["day"]["display"], "10000/250000")

    def test_get_status_unlimited(self):
        """Test get_status with unlimited limits."""
        # Setup handler with unlimited limits
        unlimited_config = {
            "req_per_min": -1,
            "req_per_day": -1,
            "tokens_per_min": -1,
            "tokens_per_day": -1
        }
        handler = RateLimitHandler(unlimited_config)
        handler.req_minute_counter = 10
        handler.req_day_counter = 100
        handler.tokens_minute_counter = 1000
        handler.tokens_day_counter = 10000

        # Get status
        status = handler.get_status()

        # Verify status
        self.assertEqual(status["requests"]["minute"]["current"], 10)
        self.assertEqual(status["requests"]["minute"]["limit"], "Unlimited")
        self.assertEqual(status["requests"]["minute"]["display"], "10/Unlimited")

        self.assertEqual(status["requests"]["day"]["current"], 100)
        self.assertEqual(status["requests"]["day"]["limit"], "Unlimited")
        self.assertEqual(status["requests"]["day"]["display"], "100/Unlimited")

        self.assertEqual(status["tokens"]["minute"]["current"], 1000)
        self.assertEqual(status["tokens"]["minute"]["limit"], "Unlimited")
        self.assertEqual(status["tokens"]["minute"]["display"], "1000/Unlimited")

        self.assertEqual(status["tokens"]["day"]["current"], 10000)
        self.assertEqual(status["tokens"]["day"]["limit"], "Unlimited")
        self.assertEqual(status["tokens"]["day"]["display"], "10000/Unlimited")

    @patch('time.time')
    def test_integration_scenario(self, mock_time):
        """Test a full integration scenario."""
        # Setup time
        initial_time = 1000.0
        mock_time.return_value = initial_time

        # Create handler with limited config
        config = {
            "req_per_min": 5,
            "req_per_day": 10,
            "tokens_per_min": 500,
            "tokens_per_day": 1000
        }
        handler = RateLimitHandler(config)

        # Make requests until we hit the minute request limit
        for i in range(5):
            can_make, reasons = handler.can_make_request(50)
            self.assertTrue(can_make)
            handler.update_counters(50)

        # Next request should fail due to minute request limit
        can_make, reasons = handler.can_make_request(50)
        self.assertFalse(can_make)
        self.assertIn("Rate limit exceeded - Minute Request Limit", reasons)

        # Advance time past 1 minute
        mock_time.return_value = initial_time + 61

        # Minute counters should reset, so we can make more requests
        can_make, reasons = handler.can_make_request(50)
        self.assertTrue(can_make)
        handler.update_counters(50)

        # But we're still limited by day counters
        # Make more requests until we hit the day request limit
        for i in range(4):
            can_make, reasons = handler.can_make_request(50)
            self.assertTrue(can_make)
            handler.update_counters(50)

        # Next request should fail due to day request limit
        can_make, reasons = handler.can_make_request(50)
        self.assertFalse(can_make)
        self.assertIn("Rate limit exceeded - Daily Request Limit", reasons)

        # Advance time past 1 day
        mock_time.return_value = initial_time + 86401

        # Day counters should reset, so we can make more requests
        can_make, reasons = handler.can_make_request(50)
        self.assertTrue(can_make)

        # Check status
        status = handler.get_status()
        self.assertEqual(status["requests"]["minute"]["current"], 0)
        self.assertEqual(status["requests"]["day"]["current"], 0)


if __name__ == "__main__":
    unittest.main()