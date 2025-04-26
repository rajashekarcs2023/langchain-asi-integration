"""Unit tests for API compatibility in the ChatASI class."""

import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_asi.chat_models import ChatASI


class TestAPICompatibility:
    """Test API compatibility and endpoint handling in the ChatASI class."""

    def setup_method(self):
        """Set up the test environment."""
        # Set up environment variables for testing
        os.environ["ASI_API_KEY"] = "test_api_key"
        
        # Create a mock response for the API
        self.mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "asi1-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response.",
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }

    def teardown_method(self):
        """Clean up after the test."""
        # Remove environment variables
        if "ASI_API_KEY" in os.environ:
            del os.environ["ASI_API_KEY"]
        if "ASI_API_BASE" in os.environ:
            del os.environ["ASI_API_BASE"]

    @patch("httpx.Client.post")
    def test_automatic_endpoint_selection_asi1(self, mock_post):
        """Test automatic endpoint selection for ASI1 models."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Initialize the chat model with an ASI1 model name
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        chat.invoke(messages)

        # Check that the API was called with the correct endpoint
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "https://api.asi1.ai/v1/chat/completions" in str(call_args)

    @patch("httpx.Client.post")
    def test_automatic_endpoint_selection_standard(self, mock_post):
        """Test automatic endpoint selection for standard ASI models."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Initialize the chat model with a standard ASI model name
        chat = ChatASI(model_name="asi-standard")  # Not an ASI1 model

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        chat.invoke(messages)

        # Check that the API was called with the correct endpoint
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "https://api.asi.ai/v1/chat/completions" in str(call_args)

    @patch("httpx.Client.post")
    def test_custom_api_base_from_env(self, mock_post):
        """Test using a custom API base URL from environment variables."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Set a custom API base URL in the environment
        os.environ["ASI_API_BASE"] = "https://custom-api.example.com/v1"

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        chat.invoke(messages)

        # Check that the API was called with the custom endpoint
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "https://custom-api.example.com/v1/chat/completions" in str(call_args)

    @patch("httpx.Client.post")
    def test_custom_api_base_from_parameter(self, mock_post):
        """Test using a custom API base URL from constructor parameter."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Initialize the chat model with a custom API base URL
        chat = ChatASI(
            model_name="asi1-mini",
            asi_api_base="https://custom-param.example.com/v1"
        )

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        chat.invoke(messages)

        # Check that the API was called with the custom endpoint
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "https://custom-param.example.com/v1/chat/completions" in str(call_args)

    @patch("httpx.Client.post")
    def test_parameter_precedence(self, mock_post):
        """Test that constructor parameter takes precedence over environment variable."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Set a custom API base URL in the environment
        os.environ["ASI_API_BASE"] = "https://env-api.example.com/v1"

        # Initialize the chat model with a different custom API base URL
        chat = ChatASI(
            model_name="asi1-mini",
            asi_api_base="https://param-api.example.com/v1"
        )

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        chat.invoke(messages)

        # Check that the API was called with the parameter endpoint (not the env one)
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "https://param-api.example.com/v1/chat/completions" in str(call_args)
        assert "https://env-api.example.com" not in str(call_args)

    @patch("httpx.Client.post")
    def test_api_key_from_env(self, mock_post):
        """Test using an API key from environment variables."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Initialize the chat model (API key is set in setup_method)
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        chat.invoke(messages)

        # Check that the API was called with the correct API key
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test_api_key"

    @patch("httpx.Client.post")
    def test_api_key_from_parameter(self, mock_post):
        """Test using an API key from constructor parameter."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Remove the API key from environment variables
        if "ASI_API_KEY" in os.environ:
            del os.environ["ASI_API_KEY"]

        # Initialize the chat model with an API key
        chat = ChatASI(model_name="asi1-mini", asi_api_key="param_api_key")

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        chat.invoke(messages)

        # Check that the API was called with the correct API key
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer param_api_key"

    @patch("httpx.Client.post")
    def test_api_key_parameter_precedence(self, mock_post):
        """Test that API key parameter takes precedence over environment variable."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Initialize the chat model with a different API key
        chat = ChatASI(model_name="asi1-mini", asi_api_key="param_api_key")

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        chat.invoke(messages)

        # Check that the API was called with the parameter API key (not the env one)
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer param_api_key"
        assert "test_api_key" not in str(call_args)

    @patch("httpx.Client.post")
    def test_request_headers(self, mock_post):
        """Test that the correct headers are sent with the request."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        chat.invoke(messages)

        # Check that the API was called with the correct headers
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"
        assert "Authorization" in headers

    @patch("httpx.Client.post")
    def test_request_payload_format(self, mock_post):
        """Test that the request payload has the correct format."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Initialize the chat model with specific parameters
        chat = ChatASI(
            model_name="asi1-mini",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )

        # Generate a response
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello, how are you?")
        ]
        chat.invoke(messages)

        # Check that the API was called with the correct payload format
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        
        # Check required fields
        assert "model" in payload
        assert payload["model"] == "asi1-mini"
        assert "messages" in payload
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert "temperature" in payload
        assert payload["temperature"] == 0.7
        assert "max_tokens" in payload
        assert payload["max_tokens"] == 100
        assert "top_p" in payload
        assert payload["top_p"] == 0.9

    @patch("httpx.Client.post")
    def test_response_processing(self, mock_post):
        """Test that the response is processed correctly."""
        # Set up a mock response with additional ASI-specific fields
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "asi1-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response.",
                        "thought": "This is my internal thought process."
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        mock_post.return_value = mock_response

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        response = chat.invoke(messages)

        # Check that the response was processed correctly
        assert isinstance(response, AIMessage)
        assert response.content == "This is a test response."
        assert "thought" in response.additional_kwargs
        assert response.additional_kwargs["thought"] == "This is my internal thought process."

    @patch("httpx.Client.post")
    def test_asi1_specific_response_format(self, mock_post):
        """Test handling of ASI1-specific response format."""
        # Set up a mock response with ASI1-specific format
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "asi1-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response.",
                        "thought": "This is my internal thought process.",
                        "additional_field": "Additional ASI1-specific field"
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "asi1_specific_field": "Some ASI1-specific data"
        }
        mock_post.return_value = mock_response

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        response = chat.invoke(messages)

        # Check that the response was processed correctly
        assert isinstance(response, AIMessage)
        assert response.content == "This is a test response."
        assert "thought" in response.additional_kwargs
        assert response.additional_kwargs["thought"] == "This is my internal thought process."
        assert "additional_field" in response.additional_kwargs
        assert response.additional_kwargs["additional_field"] == "Additional ASI1-specific field"

    @patch("httpx.Client.post")
    def test_standard_asi_response_format(self, mock_post):
        """Test handling of standard ASI response format."""
        # Set up a mock response with standard ASI format
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "asi-standard",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response."
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        }
        mock_post.return_value = mock_response

        # Initialize the chat model with a standard ASI model
        chat = ChatASI(model_name="asi-standard")

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        response = chat.invoke(messages)

        # Check that the response was processed correctly
        assert isinstance(response, AIMessage)
        assert response.content == "This is a test response."

    @patch("httpx.Client.post")
    def test_api_version_compatibility(self, mock_post):
        """Test compatibility with different API versions."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini", api_version="2023-05-15")

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        chat.invoke(messages)

        # Check that the API was called with the correct version header
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        assert "ASI-Version" in headers
        assert headers["ASI-Version"] == "2023-05-15"

    @patch("httpx.Client.post")
    def test_organization_id(self, mock_post):
        """Test using an organization ID."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Initialize the chat model with an organization ID
        chat = ChatASI(model_name="asi1-mini", organization="org-123456")

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        chat.invoke(messages)

        # Check that the API was called with the correct organization header
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        assert "ASI-Organization" in headers
        assert headers["ASI-Organization"] == "org-123456"

    @patch("httpx.Client.post")
    def test_different_completion_endpoints(self, mock_post):
        """Test compatibility with different completion endpoints."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Initialize the chat model with a custom completion path
        chat = ChatASI(
            model_name="asi1-mini",
            completion_path="/custom/completions"
        )

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        chat.invoke(messages)

        # Check that the API was called with the custom endpoint
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "https://api.asi1.ai/v1/custom/completions" in str(call_args)

    @patch("httpx.Client.post")
    def test_response_with_function_call(self, mock_post):
        """Test handling of responses with function calls (OpenAI format)."""
        # Set up a mock response with a function call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "asi1-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'll check the weather for you.",
                        "function_call": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "San Francisco"})
                        }
                    },
                    "index": 0,
                    "finish_reason": "function_call",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        }
        mock_post.return_value = mock_response

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response
        messages = [HumanMessage(content="What's the weather in San Francisco?")]
        response = chat.invoke(messages)

        # Check that the response was processed correctly
        assert isinstance(response, AIMessage)
        assert response.content == "I'll check the weather for you."
        assert "function_call" in response.additional_kwargs
        assert response.additional_kwargs["function_call"]["name"] == "get_weather"
        assert json.loads(response.additional_kwargs["function_call"]["arguments"]) == {"location": "San Francisco"}

    @patch("httpx.Client.post")
    def test_response_with_tool_calls(self, mock_post):
        """Test handling of responses with tool calls (newer OpenAI format)."""
        # Set up a mock response with tool calls
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "asi1-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'll check the weather for you.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": json.dumps({"location": "San Francisco"})
                                }
                            }
                        ]
                    },
                    "index": 0,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        }
        mock_post.return_value = mock_response

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response
        messages = [HumanMessage(content="What's the weather in San Francisco?")]
        response = chat.invoke(messages)

        # Check that the response was processed correctly
        assert isinstance(response, AIMessage)
        assert response.content == "I'll check the weather for you."
        assert hasattr(response, "tool_calls")
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "get_weather"
        assert response.tool_calls[0]["args"] == {"location": "San Francisco"}
