"""
Unit tests for the token_counter module.

Tests:
- test_count_tokens_in_message
- test_count_tokens_in_message_with_function_call
- test_count_tokens_in_message_empty_content
- test_count_tokens_in_messages_empty_list
- test_count_tokens_in_messages
- test_count_tokens_in_prompt
- test_count_tokens_in_prompt_empty
- test_count_tokens_in_request_with_messages
- test_count_tokens_in_request_with_prompt
- test_count_tokens_in_request_unknown_format
- test_estimate_completion_tokens_with_max_tokens
- test_estimate_completion_tokens_without_max_tokens
- test_estimate_completion_tokens_with_negative_max_tokens
- test_count_request_and_completion_tokens
"""

import unittest
import json
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from CutomGroqChat.token_counter import (
    count_tokens_in_message,
    count_tokens_in_messages,
    count_tokens_in_prompt,
    count_tokens_in_request,
    estimate_completion_tokens,
    count_request_and_completion_tokens
)

# Mock encoding class for testing
class MockEncoding:
    def encode(self, text):
        # Simple mock that returns 1 token per word
        if not text:
            return []
        return text.split()


class TestTokenCounter(unittest.TestCase):
    """Unit tests for the token_counter module."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_encoding = MockEncoding()
        self.model_name = "llama3-70b-8192"
        
        # Sample messages for testing
        self.system_message = {"role": "system", "content": "You are a helpful assistant."}
        self.user_message = {"role": "user", "content": "Tell me about token counting."}
        self.assistant_message = {"role": "assistant", "content": "Token counting is the process of measuring text length."}
        
        # Message with function call
        self.function_call_message = {
            "role": "assistant",
            "content": "",
            "function_call": {
                "name": "get_weather",
                "arguments": "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
            }
        }
        
        # Sample messages list
        self.messages = [
            self.system_message,
            self.user_message,
            self.assistant_message
        ]
        
        # Sample request data
        self.chat_request = {
            "model": "llama3-70b-8192",
            "messages": self.messages,
            "max_tokens": 200
        }
        
        self.prompt_request = {
            "model": "llama3-70b-8192",
            "prompt": "Generate a story about a token counter.",
            "max_tokens": 150
        }
        
        self.unknown_request = {
            "model": "llama3-70b-8192",
            "unknown_field": "This is not a standard field",
            "max_tokens": 100
        }

    @patch('tiktoken.encoding_for_model', return_value=MockEncoding())
    def test_count_tokens_in_message(self, mock_encoding_for_model):
        """Test counting tokens in a message."""
        # When using our mock encoder, it will count words as tokens
        with patch('CutomGroqChat.token_counter.DEFAULT_ENCODING', return_value=self.mock_encoding):
            result = count_tokens_in_message(self.user_message, self.mock_encoding)
            # "Tell me about token counting." = 5 words + 4 format tokens = 9 tokens
            self.assertEqual(result, 9)

    @patch('tiktoken.encoding_for_model', return_value=MockEncoding())
    def test_count_tokens_in_message_with_function_call(self, mock_encoding_for_model):
        """Test counting tokens in a message with a function call."""
        with patch('CutomGroqChat.token_counter.DEFAULT_ENCODING', return_value=self.mock_encoding):
            result = count_tokens_in_message(self.function_call_message, self.mock_encoding)
            # Empty content (0) + function call JSON (11 words as tokens) + 4 format tokens = 15 tokens
            # "name": "get_weather", "arguments": "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
            self.assertGreater(result, 4)  # Should be at least format tokens + some function tokens

    @patch('tiktoken.encoding_for_model', return_value=MockEncoding())
    def test_count_tokens_in_message_empty_content(self, mock_encoding_for_model):
        """Test counting tokens in a message with empty content."""
        with patch('CutomGroqChat.token_counter.DEFAULT_ENCODING', return_value=self.mock_encoding):
            empty_message = {"role": "user", "content": ""}
            result = count_tokens_in_message(empty_message, self.mock_encoding)
            # Empty content (0) + 4 format tokens = 4 tokens
            self.assertEqual(result, 4)

    @patch('tiktoken.encoding_for_model', return_value=MockEncoding())
    def test_count_tokens_in_messages_empty_list(self, mock_encoding_for_model):
        """Test counting tokens in an empty list of messages."""
        with patch('CutomGroqChat.token_counter.DEFAULT_ENCODING', return_value=self.mock_encoding):
            result = count_tokens_in_messages([], self.model_name)
            self.assertEqual(result, 0)

    @patch('CutomGroqChat.token_counter.count_tokens_in_message')
    def test_count_tokens_in_messages(self, mock_count_tokens_in_message):
        """Test counting tokens in a list of messages."""
        # Setup the mock to return specific values for each message
        mock_count_tokens_in_message.side_effect = [10, 15, 20]
        
        result = count_tokens_in_messages(self.messages, self.model_name)
        
        # Expected: 10 + 15 + 20 + 3 (overall formatting) = 48
        self.assertEqual(result, 48)
        self.assertEqual(mock_count_tokens_in_message.call_count, 3)

    @patch('tiktoken.encoding_for_model', return_value=MockEncoding())
    def test_count_tokens_in_prompt(self, mock_encoding_for_model):
        """Test counting tokens in a text prompt."""
        with patch('CutomGroqChat.token_counter.DEFAULT_ENCODING', self.mock_encoding):
            prompt = "This is a test prompt with eight words."
            result = count_tokens_in_prompt(prompt, self.model_name)
            # 8 words = 8 tokens with our mock
            self.assertEqual(result, 8)

    @patch('tiktoken.encoding_for_model', return_value=MockEncoding())
    def test_count_tokens_in_prompt_empty(self, mock_encoding_for_model):
        """Test counting tokens in an empty prompt."""
        with patch('CutomGroqChat.token_counter.DEFAULT_ENCODING', return_value=self.mock_encoding):
            result = count_tokens_in_prompt("", self.model_name)
            self.assertEqual(result, 0)

    @patch('CutomGroqChat.token_counter.count_tokens_in_messages')
    def test_count_tokens_in_request_with_messages(self, mock_count_tokens_in_messages):
        """Test counting tokens in a request with messages."""
        mock_count_tokens_in_messages.return_value = 100
        
        result = count_tokens_in_request(self.chat_request, self.model_name)
        
        self.assertEqual(result, 100)
        mock_count_tokens_in_messages.assert_called_once_with(self.chat_request["messages"], self.model_name)

    @patch('CutomGroqChat.token_counter.count_tokens_in_prompt')
    def test_count_tokens_in_request_with_prompt(self, mock_count_tokens_in_prompt):
        """Test counting tokens in a request with a prompt."""
        mock_count_tokens_in_prompt.return_value = 50
        
        result = count_tokens_in_request(self.prompt_request, self.model_name)
        
        self.assertEqual(result, 50)
        mock_count_tokens_in_prompt.assert_called_once_with(self.prompt_request["prompt"], self.model_name)

    def test_count_tokens_in_request_unknown_format(self):
        """Test counting tokens in a request with unknown format."""
        result = count_tokens_in_request(self.unknown_request, self.model_name)
        
        # Should return default value
        self.assertEqual(result, 10)

    def test_estimate_completion_tokens_with_max_tokens(self):
        """Test estimating completion tokens with max_tokens specified."""
        request = {"max_tokens": 200}
        result = estimate_completion_tokens(request)
        
        self.assertEqual(result, 200)

    def test_estimate_completion_tokens_without_max_tokens(self):
        """Test estimating completion tokens without max_tokens specified."""
        request = {}
        result = estimate_completion_tokens(request)
        
        # Should use default value
        self.assertEqual(result, 100)

    def test_estimate_completion_tokens_with_negative_max_tokens(self):
        """Test estimating completion tokens with negative max_tokens."""
        request = {"max_tokens": -50}
        result = estimate_completion_tokens(request)
        
        # Should use default value
        self.assertEqual(result, 100)

    @patch('CutomGroqChat.token_counter.count_tokens_in_request')
    @patch('CutomGroqChat.token_counter.estimate_completion_tokens')
    def test_count_request_and_completion_tokens(self, mock_estimate_completion_tokens, mock_count_tokens_in_request):
        """Test counting both request and completion tokens."""
        mock_count_tokens_in_request.return_value = 75
        mock_estimate_completion_tokens.return_value = 150
        
        result = count_request_and_completion_tokens(self.chat_request, self.model_name)
        
        self.assertEqual(result["prompt_tokens"], 75)
        self.assertEqual(result["completion_tokens"], 150)
        self.assertEqual(result["total_tokens"], 225)
        
        mock_count_tokens_in_request.assert_called_once_with(self.chat_request, self.model_name)
        mock_estimate_completion_tokens.assert_called_once_with(self.chat_request)


if __name__ == "__main__":
    unittest.main() 