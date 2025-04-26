"""Unit tests for streaming functionality in the ChatASI class."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from langchain_asi.chat_models import ChatASI


class TestStreaming:
    """Test the streaming functionality of the ChatASI class."""

    def setup_method(self):
        """Set up the test environment."""
        # Create a patcher for the environment validation
        self.env_patcher = patch.dict(os.environ, {"ASI_API_KEY": "fake-api-key"})
        self.env_patcher.start()
        
        # Create the chat model with the mocked environment
        self.chat = ChatASI(model_name="asi1-mini")
        
        # Mock the HTTP clients
        self.chat.client = MagicMock()
        self.chat.async_client = AsyncMock()
        
    def teardown_method(self):
        """Clean up after the test."""
        self.env_patcher.stop()

    @patch("httpx.AsyncClient.stream")
    @pytest.mark.asyncio
    async def test_astream_basic(self, mock_stream):
        """Test basic async streaming functionality."""
        # Create a mock async response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = "Streaming response"
        
        # Create an async iterator for aiter_lines
        async def mock_aiter_lines():
            lines = [
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"role":"assistant","content":"Hello"},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"content":", "},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"content":"world"},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"content":"!"},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}'
            ]
            for line in lines:
                yield line
        
        # Set up the mock to return our async iterator
        mock_response.aiter_lines = mock_aiter_lines
        mock_stream.return_value.__aenter__.return_value = mock_response

        # Create a streaming model
        streaming_chat = self.chat.with_config({"streaming": True})

        # Generate a streaming response
        messages = [HumanMessage(content="Say hello")]
        chunks = []
        async for chunk in streaming_chat.astream(messages):
            chunks.append(chunk)

        # Check that the chunks were received correctly
        assert len(chunks) == 4  # 4 content chunks
        assert chunks[0].content == "Hello"
        assert chunks[1].content == ", "
        assert chunks[2].content == "world"
        assert chunks[3].content == "!"
        
        # Combine all chunks to get the full message
        full_content = "".join(chunk.content for chunk in chunks)
        assert full_content == "Hello, world!"

    @patch("httpx.AsyncClient.stream")
    @pytest.mark.asyncio
    async def test_astream_tool_calls(self, mock_stream):
        """Test streaming with tool calls."""
        # Create a mock async response with tool calls
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = "Streaming response with tool calls"
        
        # Create an async iterator for aiter_lines
        async def mock_aiter_lines():
            lines = [
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"role":"assistant","content":"I\'ll check the weather"},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather"}}]},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"arguments":"{\\"}}]},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"arguments":"\\"location\\":\\""}}]},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"arguments":"San Francisco\\""}}]},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"arguments":"}"}}]},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{},"index":0,"finish_reason":"tool_calls"}]}'
            ]
            for line in lines:
                yield line
        
        # Set up the mock to return our async iterator
        mock_response.aiter_lines = mock_aiter_lines
        mock_stream.return_value.__aenter__.return_value = mock_response

        # Create a streaming model
        streaming_chat = self.chat.with_config({"streaming": True})

        # Generate a streaming response
        messages = [HumanMessage(content="What's the weather in San Francisco?")]
        chunks = []
        async for chunk in streaming_chat.astream(messages):
            chunks.append(chunk)

        # Check that the chunks were received correctly
        assert len(chunks) > 0
        
        # The last chunk should have the complete tool call
        last_chunk = chunks[-1]
        assert hasattr(last_chunk, "tool_calls")
        assert len(last_chunk.tool_calls) == 1
        assert last_chunk.tool_calls[0]["name"] == "get_weather"
        assert last_chunk.tool_calls[0]["args"] == {"location": "San Francisco"}

    @patch("httpx.AsyncClient.stream")
    @pytest.mark.asyncio
    async def test_astream_error_handling(self, mock_stream):
        """Test error handling during streaming."""
        # Create a mock async response with an error
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Test streaming error"
        
        # Create an async iterator that raises an exception
        async def mock_aiter_lines():
            raise Exception("Test streaming error")
        
        # Set up the mock to return our async iterator
        mock_response.aiter_lines = mock_aiter_lines
        mock_stream.return_value.__aenter__.return_value = mock_response

        # Create a streaming model
        streaming_chat = self.chat.with_config({"streaming": True})

        # Generate a streaming response and expect an error
        messages = [HumanMessage(content="Say hello")]
        with pytest.raises(Exception) as excinfo:
            async for chunk in streaming_chat.astream(messages):
                pass

        # Check that the error message is correct
        assert "Test streaming error" in str(excinfo.value)

    @patch("httpx.AsyncClient.stream")
    @pytest.mark.asyncio
    async def test_astream_malformed_json(self, mock_stream):
        """Test handling of malformed JSON during streaming."""
        # Create a mock async response with malformed JSON
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = "Streaming response with malformed JSON"
        
        # Create an async iterator for aiter_lines
        async def mock_aiter_lines():
            lines = [
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"role":"assistant","content":"Hello"},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"content":", "},"index":0,"finish_reason":null}]}',
                '{malformed_json}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"content":"world"},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}'
            ]
            for line in lines:
                yield line
        
        # Set up the mock to return our async iterator
        mock_response.aiter_lines = mock_aiter_lines
        mock_stream.return_value.__aenter__.return_value = mock_response

        # Create a streaming model
        streaming_chat = self.chat.with_config({"streaming": True})

        # Generate a streaming response
        messages = [HumanMessage(content="Say hello")]
        chunks = []
        async for chunk in streaming_chat.astream(messages):
            chunks.append(chunk)

        # Check that valid chunks were still processed
        assert len(chunks) > 0
        assert chunks[0].content == "Hello"
        assert "world" in chunks[-1].content

    @patch("langchain_asi.chat_models.ChatASI._stream")
    def test_stream_sync(self, mock_stream_method):
        """Test synchronous streaming functionality."""
        # Create mock chunks to return
        hello_chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="Hello"),
            generation_info={"finish_reason": None}
        )
        world_chunk = ChatGenerationChunk(
            message=AIMessageChunk(content=", world!"),
            generation_info={"finish_reason": None}
        )
        
        # Set up the mock to return our chunks
        mock_stream_method.return_value = [hello_chunk, world_chunk]

        # Create a streaming model
        streaming_chat = self.chat.with_config({"streaming": True})

        # Generate a streaming response
        messages = [HumanMessage(content="Say hello")]
        chunks = []
        for chunk in streaming_chat.stream(messages):
            chunks.append(chunk)

        # Check that the chunks were received correctly
        assert len(chunks) == 2  # 2 content chunks
        assert chunks[0].content == "Hello"
        assert chunks[1].content == ", world!"
        
        # Combine all chunks to get the full message
        full_content = "".join(chunk.content for chunk in chunks)
        assert full_content == "Hello, world!"

    @patch("httpx.AsyncClient.stream")
    @pytest.mark.asyncio
    async def test_astream_thought_field(self, mock_stream):
        """Test streaming with ASI-specific 'thought' field."""
        # Create a mock async response with a thought field
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = "Streaming response with thought field"
        
        # Create an async iterator for aiter_lines
        async def mock_aiter_lines():
            lines = [
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"role":"assistant","content":"Here\'s my answer","thought":"I need to think about this"},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"content":".", "thought":" carefully."},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}'
            ]
            for line in lines:
                yield line
        
        # Set up the mock to return our async iterator
        mock_response.aiter_lines = mock_aiter_lines
        mock_stream.return_value.__aenter__.return_value = mock_response

        # Create a streaming model
        streaming_chat = self.chat.with_config({"streaming": True})

        # Generate a streaming response
        messages = [HumanMessage(content="Give me a thoughtful answer")]
        chunks = []
        async for chunk in streaming_chat.astream(messages):
            chunks.append(chunk)

        # Check that the chunks were received correctly
        assert len(chunks) == 2  # 2 content chunks
        assert chunks[0].content == "Here's my answer"
        assert chunks[1].content == "."
        
        # Check that the thought field was processed
        assert hasattr(chunks[0], "additional_kwargs")
        assert "thought" in chunks[0].additional_kwargs
        assert chunks[0].additional_kwargs["thought"] == "I need to think about this"
        assert chunks[1].additional_kwargs["thought"] == " carefully."

    @patch("httpx.AsyncClient.stream")
    @pytest.mark.asyncio
    async def test_astream_callback(self, mock_stream):
        """Test streaming with callbacks."""
        # Create a mock async response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = "Streaming response with callbacks"
        
        # Create an async iterator for aiter_lines
        async def mock_aiter_lines():
            lines = [
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"role":"assistant","content":"Hello"},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"content":", world!"},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}'
            ]
            for line in lines:
                yield line
        
        # Set up the mock to return our async iterator
        mock_response.aiter_lines = mock_aiter_lines
        mock_stream.return_value.__aenter__.return_value = mock_response

        # Create a callback handler
        class TestCallbackHandler:
            def __init__(self):
                self.tokens = []
                self.run_inline = True  # Required for inline execution
                self.ignore_chat_model = False  # Required for chat model callbacks
                self.ignore_llm = False  # Required for LLM callbacks
                self.raise_error = False  # Required for error handling
                
            def on_llm_new_token(self, token, **kwargs):
                self.tokens.append(token)
                
            def on_chat_model_start(self, *args, **kwargs):
                # Implement this to handle chat model start events
                pass

        callback_handler = TestCallbackHandler()

        # Create a streaming model with the callback
        streaming_chat = self.chat.with_config({
            "streaming": True,
            "callbacks": [callback_handler]
        })

        # Generate a streaming response
        messages = [HumanMessage(content="Say hello")]
        final_response = None
        async for chunk in streaming_chat.astream(messages):
            final_response = chunk

        # Check that the callback received the tokens
        assert len(callback_handler.tokens) == 2
        assert callback_handler.tokens[0] == "Hello"
        assert callback_handler.tokens[1] == ", world!"
        
        # Check that the final response contains the last chunk
        # Note: In streaming, each chunk only contains the delta, not the accumulated content
        assert final_response.content == ", world!"

    @patch("httpx.AsyncClient.stream")
    @pytest.mark.asyncio
    async def test_astream_empty_response(self, mock_stream):
        """Test streaming with an empty response."""
        # Create a mock async response with empty content
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = "Streaming response with empty content"
        
        # Create an async iterator for aiter_lines
        async def mock_aiter_lines():
            lines = [
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{"role":"assistant","content":""},"index":0,"finish_reason":null}]}',
                '{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"asi1-mini","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}'
            ]
            for line in lines:
                yield line
        
        # Set up the mock to return our async iterator
        mock_response.aiter_lines = mock_aiter_lines
        mock_stream.return_value.__aenter__.return_value = mock_response

        # Create a streaming model
        streaming_chat = self.chat.with_config({"streaming": True})

        # Generate a streaming response
        messages = [HumanMessage(content="Say hello")]
        chunks = []
        async for chunk in streaming_chat.astream(messages):
            chunks.append(chunk)

        # Check that we got a response with empty content
        assert len(chunks) == 1
        assert chunks[0].content == ""
