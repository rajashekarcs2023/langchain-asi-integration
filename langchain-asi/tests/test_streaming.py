"""Tests for the ASI1ChatModel streaming functionality."""
import os
import unittest
from unittest import mock

from langchain_asi import ASI1ChatModel
from langchain_core.messages import HumanMessage


class TestASI1ChatModelStreaming(unittest.TestCase):
    """Test ASI1ChatModel streaming functionality."""

    def setUp(self):
        """Set up the test."""
        # Mocking response for requests post
        self.mock_response = mock.MagicMock()
        self.mock_response.iter_lines.return_value = [
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"This"},"finish_reason":null}]}',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}]}',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"index":0,"delta":{"content":" a"},"finish_reason":null}]}',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"index":0,"delta":{"content":" test"},"finish_reason":null}]}',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"index":0,"delta":{"content":" response."},"finish_reason":null}]}',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            b'data: [DONE]'
        ]
        self.mock_response.raise_for_status = mock.MagicMock()
        
    @mock.patch("requests.post")
    def test_streaming(self, mock_post):
        """Test streaming functionality."""
        # Configure the mock
        mock_post.return_value = self.mock_response
        
        # Create the ASI1ChatModel with streaming enabled
        chat = ASI1ChatModel(
            asi1_api_key="fake-api-key",
            model_name="asi1-mini",
            streaming=True
        )
        
        # Create a sample message
        messages = [HumanMessage(content="Hello, how are you?")]
        
        # Test stream method
        chunks = list(chat.stream(messages))
        
        # Check that we got the expected number of chunks
        self.assertEqual(len(chunks), 6)
        
        # Check that the chunks contain the expected content
        self.assertEqual(chunks[0].content, "This")
        self.assertEqual(chunks[1].content, " is")
        self.assertEqual(chunks[2].content, " a")
        self.assertEqual(chunks[3].content, " test")
        self.assertEqual(chunks[4].content, " response.")
        self.assertEqual(chunks[5].content, "")
        
        # Check that the API was called with the right parameters
        mock_post.assert_called_once()
        url_called = mock_post.call_args[0][0]
        self.assertEqual(url_called, "https://api.asi1.ai/v1/chat/completions")
        
        # Check the JSON payload
        json_payload = mock_post.call_args[1]["json"]
        self.assertEqual(json_payload["model"], "asi1-mini")
        self.assertEqual(json_payload["messages"][0]["role"], "user")
        self.assertEqual(json_payload["messages"][0]["content"], "Hello, how are you?")
        self.assertTrue(json_payload["stream"])


if __name__ == "__main__":
    unittest.main()
