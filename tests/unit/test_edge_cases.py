"""Unit tests for edge cases and error handling in the ChatASI class."""

import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_asi.chat_models import ChatASI


class TestEdgeCases:
    """Test edge cases and error handling in the ChatASI class."""

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

    @patch("httpx.Client.post")
    def test_missing_api_key(self, mock_post):
        """Test behavior when API key is missing."""
        # Remove the API key from environment variables
        if "ASI_API_KEY" in os.environ:
            del os.environ["ASI_API_KEY"]
        
        # Attempt to initialize the chat model without an API key
        with pytest.raises(ValueError) as excinfo:
            ChatASI(model_name="asi1-mini")
        
        # Check that the error message mentions the API key
        assert "API key" in str(excinfo.value)

    @patch("httpx.Client.post")
    def test_invalid_model_name(self, mock_post):
        """Test behavior with an invalid model name."""
        # Set up the mock response with an error
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Model not found"
        mock_response.json.return_value = {"error": {"message": "Model not found"}}
        mock_response.request = MagicMock()
        mock_post.return_value = mock_response

        # Initialize the chat model with an invalid model name
        chat = ChatASI(model_name="nonexistent-model")

        # Generate a response and expect an error
        messages = [HumanMessage(content="Hello, how are you?")]
        with pytest.raises(ValueError) as excinfo:
            chat.invoke(messages)

        # Check that the error message is correct
        assert "Model not found" in str(excinfo.value)

    @patch("httpx.Client.post")
    def test_rate_limit_error(self, mock_post):
        """Test behavior when hitting a rate limit."""
        # Set up the mock response with a rate limit error
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
        mock_response.request = MagicMock()
        mock_post.return_value = mock_response

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response and expect a rate limit error
        messages = [HumanMessage(content="Hello, how are you?")]
        with pytest.raises(ValueError) as excinfo:
            chat.invoke(messages)

        # Check that the error message is correct
        assert "Rate limit exceeded" in str(excinfo.value)

    @patch("httpx.Client.post")
    def test_server_error(self, mock_post):
        """Test behavior when the server returns an error."""
        # Set up the mock response with a server error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_response.json.return_value = {"error": {"message": "Internal server error"}}
        mock_response.request = MagicMock()
        mock_post.return_value = mock_response

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response and expect a server error
        messages = [HumanMessage(content="Hello, how are you?")]
        with pytest.raises(ValueError) as excinfo:
            chat.invoke(messages)

        # Check that the error message is correct
        assert "Internal server error" in str(excinfo.value)

    @patch("httpx.Client.post")
    def test_empty_messages(self, mock_post):
        """Test behavior with empty messages."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response with empty messages
        messages = []
        with pytest.raises(ValueError) as excinfo:
            chat.invoke(messages)

        # Check that the error message is correct
        assert "messages" in str(excinfo.value)

    def test_invalid_message_type(self):
        """Test behavior with invalid message types."""
        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Test direct validation in _convert_input_to_messages
        with pytest.raises(TypeError) as excinfo:
            # Call the method directly with an invalid message type
            chat._convert_input_to_messages([{"invalid": "message"}])

        # Check that the error message is correct
        assert "message" in str(excinfo.value)

    @patch("httpx.Client.post")
    def test_malformed_response(self, mock_post):
        """Test behavior with a malformed response."""
        # Set up the mock response with a malformed structure
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "asi1-mini",
            # Missing the 'choices' field
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        mock_post.return_value = mock_response

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response and expect a ValueError for invalid response format
        messages = [HumanMessage(content="Hello, how are you?")]
        with pytest.raises(ValueError) as excinfo:
            chat.invoke(messages)
            
        # Check that the error message mentions 'Invalid response format'
        assert "Invalid response format" in str(excinfo.value)
        
        # Check that the response data is included in the error message
        assert str(mock_response.json.return_value) in str(excinfo.value)

    @patch("httpx.Client.post")
    def test_empty_response_content(self, mock_post):
        """Test behavior with empty response content."""
        # Set up the mock response with empty content
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
                        "content": "",  # Empty content
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 0,
                "total_tokens": 10,
            },
        }
        mock_post.return_value = mock_response

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response
        messages = [HumanMessage(content="Hello, how are you?")]
        response = chat.invoke(messages)

        # Check that the response is an AIMessage with empty content
        assert isinstance(response, AIMessage)
        assert response.content == ""

    @patch("httpx.Client.post")
    def test_tool_call_without_arguments(self, mock_post):
        """Test behavior with a tool call that has no arguments."""
        # Set up the mock response with a tool call without arguments
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
                                    # Missing arguments
                                }
                            }
                        ]
                    },
                    "index": 0,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40,
            },
        }
        mock_post.return_value = mock_response

        # Define a tool
        @tool
        def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return f"The weather in {location} is sunny."

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Bind the tool to the model
        chat_with_tools = chat.bind_tools([get_weather])

        # Generate a response
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What's the weather in San Francisco?"),
        ]
        
        # This should handle the missing arguments gracefully
        response = chat_with_tools.invoke(messages)
        
        # Check that the response is an AIMessage with tool calls
        assert isinstance(response, AIMessage)
        assert response.content == "I'll check the weather for you."
        assert hasattr(response, "tool_calls")
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "get_weather"
        assert response.tool_calls[0]["args"] == {}  # Empty arguments

    @patch("httpx.Client.post")
    def test_tool_call_with_invalid_arguments(self, mock_post):
        """Test behavior with a tool call that has invalid arguments."""
        # Set up the mock response with a tool call with invalid arguments
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
                                    "arguments": "This is not valid JSON"
                                }
                            }
                        ]
                    },
                    "index": 0,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40,
            },
        }
        mock_post.return_value = mock_response

        # Define a tool
        @tool
        def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return f"The weather in {location} is sunny."

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Bind the tool to the model
        chat_with_tools = chat.bind_tools([get_weather])

        # Generate a response
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What's the weather in San Francisco?"),
        ]
        
        # This should handle the invalid arguments gracefully
        response = chat_with_tools.invoke(messages)
        
        # Check that the response is an AIMessage with tool calls
        assert isinstance(response, AIMessage)
        assert response.content == "I'll check the weather for you."
        assert hasattr(response, "tool_calls")
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "get_weather"
        assert response.tool_calls[0]["args"] == {}  # Empty arguments due to invalid JSON

    @patch("httpx.Client.post")
    def test_tool_call_with_unknown_tool(self, mock_post):
        """Test behavior with a tool call for an unknown tool."""
        # Set up the mock response with a tool call for an unknown tool
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
                                    "name": "unknown_tool",  # Unknown tool
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
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40,
            },
        }
        mock_post.return_value = mock_response

        # Define a tool
        @tool
        def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return f"The weather in {location} is sunny."

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Bind the tool to the model
        chat_with_tools = chat.bind_tools([get_weather])

        # Generate a response
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What's the weather in San Francisco?"),
        ]
        
        # This should handle the unknown tool gracefully
        response = chat_with_tools.invoke(messages)
        
        # Check that the response is an AIMessage with tool calls
        assert isinstance(response, AIMessage)
        assert response.content == "I'll check the weather for you."
        assert hasattr(response, "tool_calls")
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "unknown_tool"
        assert response.tool_calls[0]["args"] == {"location": "San Francisco"}

    @patch("httpx.Client.post")
    def test_timeout_error(self, mock_post):
        """Test behavior when a request times out."""
        # Set up the mock to raise a timeout error
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        # Initialize the chat model with a short timeout
        chat = ChatASI(model_name="asi1-mini", request_timeout=1.0)

        # Generate a response and expect a timeout error
        messages = [HumanMessage(content="Hello, how are you?")]
        with pytest.raises(httpx.TimeoutException) as excinfo:
            chat.invoke(messages)

        # Check that the error message is correct
        assert "timed out" in str(excinfo.value)

    @patch("httpx.Client.post")
    def test_connection_error(self, mock_post):
        """Test behavior when there's a connection error."""
        # Set up the mock to raise a connection error
        mock_post.side_effect = httpx.ConnectError("Connection error")

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response and expect a connection error
        messages = [HumanMessage(content="Hello, how are you?")]
        with pytest.raises(httpx.ConnectError) as excinfo:
            chat.invoke(messages)

        # Check that the error message is correct
        assert "Connection error" in str(excinfo.value)

    @patch("httpx.Client.post")
    def test_invalid_parameter_values(self, mock_post):
        """Test behavior with invalid parameter values."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Test with invalid temperature (too high)
        with pytest.raises(ValueError) as excinfo:
            ChatASI(model_name="asi1-mini", temperature=2.1)
        # The error message format changed with Pydantic v2, so we check for 'Temperature' (capitalized)
        assert "Temperature" in str(excinfo.value)

        # Test with invalid temperature (too low)
        with pytest.raises(ValueError) as excinfo:
            ChatASI(model_name="asi1-mini", temperature=-0.1)
        assert "Temperature" in str(excinfo.value)
        
        # We've successfully validated temperature parameters

        # Test with invalid top_p (too high)
        with pytest.raises(ValueError) as excinfo:
            ChatASI(model_name="asi1-mini", top_p=2.0)
        # The error message format changed with Pydantic v2, so we check for 'Top_p' (capitalized)
        assert "Top_p" in str(excinfo.value)

        # Test with invalid top_p (too low)
        with pytest.raises(ValueError) as excinfo:
            ChatASI(model_name="asi1-mini", top_p=0)
        assert "Top_p" in str(excinfo.value)

    @patch("httpx.Client.post")
    def test_extremely_long_message(self, mock_post):
        """Test behavior with an extremely long message."""
        # Set up the mock response with a token limit error
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Token limit exceeded"
        mock_response.json.return_value = {
            "error": {
                "message": "This model's maximum context length is 8192 tokens, however you requested 10000 tokens. Please reduce the length of your prompt."
            }
        }
        mock_response.request = MagicMock()
        mock_post.return_value = mock_response

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response with an extremely long message
        long_message = "This is a very long message. " * 1000  # Approximately 6000 tokens
        messages = [HumanMessage(content=long_message)]
        
        # This should raise an error about the token limit
        with pytest.raises(ValueError) as excinfo:
            chat.invoke(messages)

        # Check that the error message mentions the token limit
        assert "context length" in str(excinfo.value) or "token" in str(excinfo.value)

    @patch("httpx.Client.post")
    def test_multiple_system_messages(self, mock_post):
        """Test behavior with multiple system messages."""
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value = mock_response

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Generate a response with multiple system messages
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            SystemMessage(content="You should be concise."),
            HumanMessage(content="Hello, how are you?"),
        ]
        
        # This should combine the system messages
        chat.invoke(messages)
        
        # Check that the API was called with the combined system messages
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        request_data = call_args[1]["json"]
        
        # Check that there's only one system message
        system_messages = [msg for msg in request_data["messages"] if msg["role"] == "system"]
        assert len(system_messages) == 1
        
        # Check that the system message contains both original messages
        assert "You are a helpful assistant." in system_messages[0]["content"]
        assert "You should be concise." in system_messages[0]["content"]

    @patch("httpx.Client.post")
    def test_with_structured_output_dict_schema(self, mock_post):
        """Test with_structured_output using a dictionary schema."""
        # Define a schema as a dictionary
        dict_schema = {
            "title": "Person",
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The person's name"},
                "age": {"type": "integer", "description": "The person's age"}
            },
            "required": ["name", "age"]
        }
        
        # Mock the response
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
                        "content": "I'll provide information about John Doe.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "Person",
                                    "arguments": json.dumps({
                                        "name": "John Doe",
                                        "age": 30
                                    })
                                }
                            }
                        ]
                    },
                    "index": 0,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40,
            },
        }
        mock_post.return_value = mock_response

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Create a model with structured output using a dictionary schema
        structured_model = chat.with_structured_output(
            dict_schema,
            method="function_calling"
        )

        # Generate a response
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Tell me about John Doe")
        ]
        
        result = structured_model.invoke(messages)
        
        # Check that the result is a dictionary with the expected structure
        assert isinstance(result, dict)
        assert result["name"] == "John Doe"
        assert result["age"] == 30

    @patch("httpx.Client.post")
    def test_invalid_api_base_url(self, mock_post):
        """Test behavior with an invalid API base URL."""
        # Set up the mock to raise a connection error
        mock_post.side_effect = httpx.ConnectError("Invalid URL")

        # Initialize the chat model with an invalid API base URL
        chat = ChatASI(model_name="asi1-mini", asi_api_base="https://invalid-url.example.com")

        # Generate a response and expect a connection error
        messages = [HumanMessage(content="Hello, how are you?")]
        with pytest.raises(httpx.ConnectError) as excinfo:
            chat.invoke(messages)

        # Check that the error message is correct
        assert "Invalid URL" in str(excinfo.value)

    @patch("httpx.Client.post")
    def test_tool_choice_none(self, mock_post):
        """Test binding tools with tool_choice=none."""
        # Set up the mock response
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
                        "content": "The weather in San Francisco is typically mild with fog in the mornings.",
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40,
            },
        }
        mock_post.return_value = mock_response

        # Define a tool
        @tool
        def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return f"The weather in {location} is sunny."

        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")

        # Bind the tool with tool_choice=none
        chat_with_tools = chat.bind_tools([get_weather], tool_choice="none")

        # Generate a response
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What's the weather in San Francisco?"),
        ]
        
        response = chat_with_tools.invoke(messages)
        
        # Check that the API was called with tool_choice=none
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        request_data = call_args[1]["json"]
        assert "tool_choice" in request_data
        assert request_data["tool_choice"] == "none"
        
        # Check that the response doesn't have tool calls
        assert isinstance(response, AIMessage)
        assert response.content == "The weather in San Francisco is typically mild with fog in the mornings."
        assert not hasattr(response, "tool_calls") or not response.tool_calls
