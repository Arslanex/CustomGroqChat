"""
Unit tests for the APIClient class in the CutomGroqChat module.

Tests:
- test_initialization
- test_get_session_creates_new_session
- test_get_session_reuses_existing_session
- test_get_session_replaces_closed_session
- test_close_session
- test_close_nonexistent_session
- test_close_already_closed_session
- test_post_request_success
- test_api_error_handling
- test_invalid_json_response
- test_client_error_handling
- test_response_processing
- test_empty_payload
- test_timeout_handling
- test_complete_flow
"""


import unittest
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import aiohttp
from aiohttp import ClientResponse

from CutomGroqChat.api_client import APIClient
from CutomGroqChat.exceptions import APICallException


class APIClientTests(unittest.IsolatedAsyncioTestCase):
    """Test cases for the APIClient class."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_url = "https://api.groq.com"
        self.api_key = "test-api-key"
        self.client = APIClient(self.base_url, self.api_key)
        
        # Standard test data
        self.test_endpoint = "chat/completions"
        self.test_payload = {"model": "llama3-70b-8192", "messages": [{"role": "user", "content": "Hello"}]}
        self.test_response = {"id": "test-id", "choices": [{"message": {"content": "Hi there!"}}]}

    async def asyncTearDown(self):
        """Clean up after tests."""
        if self.client.session and not self.client.session.closed:
            try:
                await self.client.close()
            except (TypeError, AttributeError):
                # Handle case where session is a mock without required async methods
                pass

    # 1. Initialization Tests
    def test_initialization(self):
        """Test that client initializes with correct parameters."""
        self.assertEqual(self.client.base_url, self.base_url)
        self.assertEqual(self.client.api_key, self.api_key)
        self.assertIsNone(self.client.session)

    # 2. Session Management Tests
    @patch('aiohttp.ClientSession')
    async def test_get_session_creates_new_session(self, mock_session):
        """Test that _get_session creates a new session when none exists."""
        # Use AsyncMock for the session to support await operations
        mock_session.return_value = AsyncMock()
        
        session = await self.client._get_session()
        
        mock_session.assert_called_once()
        self.assertEqual(session, self.client.session)
        
        # Verify correct headers
        expected_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        mock_session.assert_called_with(headers=expected_headers)

    @patch('aiohttp.ClientSession')
    async def test_get_session_reuses_existing_session(self, mock_session):
        """Test that _get_session reuses an existing session."""
        # Use AsyncMock to support await operations
        mock_existing_session = AsyncMock()
        mock_existing_session.closed = False
        self.client.session = mock_existing_session
        
        session = await self.client._get_session()
        
        mock_session.assert_not_called()
        self.assertEqual(session, mock_existing_session)

    @patch('aiohttp.ClientSession')
    async def test_get_session_replaces_closed_session(self, mock_session):
        """Test that _get_session replaces a closed session."""
        mock_closed_session = AsyncMock()
        mock_closed_session.closed = True
        self.client.session = mock_closed_session
        
        mock_new_session = AsyncMock()
        mock_session.return_value = mock_new_session
        
        session = await self.client._get_session()
        
        mock_session.assert_called_once()
        self.assertEqual(session, mock_new_session)

    async def test_close_session(self):
        """Test that close() properly closes the session."""
        mock_session = AsyncMock()
        mock_session.closed = False
        self.client.session = mock_session
        
        await self.client.close()
        
        mock_session.close.assert_called_once()

    async def test_close_nonexistent_session(self):
        """Test that close() handles a nonexistent session gracefully."""
        self.client.session = None
        
        # This should not raise any exceptions
        await self.client.close()

    async def test_close_already_closed_session(self):
        """Test that close() handles an already closed session gracefully."""
        mock_session = AsyncMock()  # Use AsyncMock to support close()
        mock_session.closed = True
        self.client.session = mock_session
        
        # This should not raise any exceptions or call close() again
        await self.client.close()
        mock_session.close.assert_not_called()

    # 3. POST Request Tests
    @patch('aiohttp.ClientSession.post')
    async def test_post_request_success(self, mock_post):
        """Test successful POST request."""
        # Mock the response
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=json.dumps(self.test_response))
        mock_post.return_value.__aenter__.return_value = mock_response
        
        response = await self.client.post_request(self.test_endpoint, self.test_payload)
        
        # Verify the request was made with correct parameters
        expected_url = f"{self.base_url}/{self.test_endpoint}"
        mock_post.assert_called_with(expected_url, json=self.test_payload)
        
        # Verify the response was processed correctly
        self.assertEqual(response, self.test_response)

    # 4. Error Handling Tests
    @patch('aiohttp.ClientSession.post')
    async def test_api_error_handling(self, mock_post):
        """Test handling of API errors."""
        error_response = {"error": {"message": "API error", "type": "invalid_request_error"}}
        
        # Mock a 400 error response
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value=json.dumps(error_response))
        mock_post.return_value.__aenter__.return_value = mock_response
        
        with self.assertRaises(APICallException) as context:
            await self.client.post_request(self.test_endpoint, self.test_payload)
        
        # Verify exception contains correct information
        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("API call failed with status 400", context.exception.message)

    @patch('aiohttp.ClientSession.post')
    async def test_invalid_json_response(self, mock_post):
        """Test handling of invalid JSON responses."""
        # Mock an invalid JSON response
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="Not a JSON response")
        mock_post.return_value.__aenter__.return_value = mock_response
        
        with self.assertRaises(APICallException) as context:
            await self.client.post_request(self.test_endpoint, self.test_payload)
        
        # Verify exception contains correct information
        self.assertEqual(context.exception.status_code, 200)
        self.assertIn("Failed to decode JSON response", context.exception.message)

    @patch('aiohttp.ClientSession.post')
    async def test_client_error_handling(self, mock_post):
        """Test handling of client errors like network issues."""
        # Mock a network error
        mock_post.side_effect = aiohttp.ClientError("Network error")
        
        with self.assertRaises(APICallException) as context:
            await self.client.post_request(self.test_endpoint, self.test_payload)
        
        # Verify exception contains correct information
        self.assertEqual(context.exception.status_code, 500)
        self.assertIn("Client error: Network error", context.exception.message)

    # 5. Response Processing Tests
    @patch('aiohttp.ClientSession.post')
    async def test_response_processing(self, mock_post):
        """Test processing of different response structures."""
        # Test with a complex nested response
        complex_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-3.5-turbo-0301",
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 7,
                "total_tokens": 20
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?"
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ]
        }
        
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=json.dumps(complex_response))
        mock_post.return_value.__aenter__.return_value = mock_response
        
        response = await self.client.post_request(self.test_endpoint, self.test_payload)
        
        # Verify the complex response was processed correctly
        self.assertEqual(response, complex_response)
        self.assertEqual(response["choices"][0]["message"]["content"], 
                         "Hello! How can I help you today?")

    # 8. Edge Cases
    @patch('aiohttp.ClientSession.post')
    async def test_empty_payload(self, mock_post):
        """Test with an empty payload."""
        empty_payload = {}
        
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=json.dumps({"result": "success"}))
        mock_post.return_value.__aenter__.return_value = mock_response
        
        response = await self.client.post_request(self.test_endpoint, empty_payload)
        
        # Verify the request was made with the empty payload
        mock_post.assert_called_with(f"{self.base_url}/{self.test_endpoint}", json=empty_payload)
        self.assertEqual(response, {"result": "success"})

    @patch('aiohttp.ClientSession.post')
    async def test_timeout_handling(self, mock_post):
        """Test handling of timeout errors."""
        # Correctly mock a ClientError subclass
        mock_post.side_effect = aiohttp.ClientError("Request timed out")
        
        with self.assertRaises(APICallException) as context:
            await self.client.post_request(self.test_endpoint, self.test_payload)
        
        # Verify exception contains correct information
        self.assertEqual(context.exception.status_code, 500)
        self.assertIn("Client error: Request timed out", context.exception.message)

    # 7. Integration test
    @patch('aiohttp.ClientSession.post')
    async def test_complete_flow(self, mock_post):
        """Test the complete flow from session creation to response processing."""
        # Mock the session creation and response
        mock_response = AsyncMock(spec=ClientResponse)
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=json.dumps(self.test_response))
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Initial state check
        self.assertIsNone(self.client.session)
        
        # Execute the request
        response = await self.client.post_request(self.test_endpoint, self.test_payload)
        
        # Verify session was created
        self.assertIsNotNone(self.client.session)
        
        # Verify correct response
        self.assertEqual(response, self.test_response)
        
        # Close the session
        await self.client.close()


if __name__ == "__main__":
    unittest.main() 