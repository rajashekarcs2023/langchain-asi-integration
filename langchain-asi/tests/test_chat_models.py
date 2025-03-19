"""Tests for the ASI1ChatModel."""
import os
import unittest
from unittest import mock

from langchain_asi import ASI1ChatModel, ASIJsonOutputParserWithValidation
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List


class TestASI1ChatModel(unittest.TestCase):
    """Test ASI1ChatModel."""

    def setUp(self):
        """Set up the test."""
        # Mocking response for requests post
        self.mock_response = mock.MagicMock()
        self.mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "asi1-mini",
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 7,
                "total_tokens": 20
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response."
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ]
        }
        self.mock_response.raise_for_status = mock.MagicMock()
        
    @mock.patch("requests.post")
    def test_chat_completion(self, mock_post):
        """Test chat completion."""
        # Configure the mock
        mock_post.return_value = self.mock_response
        
        # Create the ASI1ChatModel
        chat = ASI1ChatModel(
            asi1_api_key="fake-api-key",
            model_name="asi1-mini"
        )
        
        # Create a sample message
        messages = [HumanMessage(content="Hello, how are you?")]
        
        # Test invoke method
        response = chat.invoke(messages)
        
        # Check that the response is as expected
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(response.content, "This is a test response.")
        
        # Check that the API was called with the right parameters
        mock_post.assert_called_once()
        url_called = mock_post.call_args[0][0]
        self.assertEqual(url_called, "https://api.asi1.ai/v1/chat/completions")
        
        # Check the JSON payload
        json_payload = mock_post.call_args[1]["json"]
        self.assertEqual(json_payload["model"], "asi1-mini")
        self.assertEqual(json_payload["messages"][0]["role"], "user")
        self.assertEqual(json_payload["messages"][0]["content"], "Hello, how are you?")
    
    @mock.patch("requests.post")
    def test_with_system_message(self, mock_post):
        """Test with system message."""
        # Configure the mock
        mock_post.return_value = self.mock_response
        
        # Create the ASI1ChatModel
        chat = ASI1ChatModel(
            asi1_api_key="fake-api-key",
            model_name="asi1-mini"
        )
        
        # Create messages with a system message
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello, how are you?")
        ]
        
        # Test invoke method
        chat.invoke(messages)
        
        # Check the JSON payload
        json_payload = mock_post.call_args[1]["json"]
        self.assertEqual(json_payload["messages"][0]["role"], "system")
        self.assertEqual(json_payload["messages"][0]["content"], "You are a helpful assistant.")
    
    @mock.patch("langchain_asi.chat_models.requests.post")
    def test_bind_tools(self, mock_post):
        """Test binding tools to the model."""
        # Configure the mock
        mock_post.return_value = self.mock_response
        
        # Create the ASI1ChatModel
        chat = ASI1ChatModel(
            asi1_api_key="fake-api-key",
            model_name="asi1-mini"
        )
        
        # Define a tool dictionary
        tool = {
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            },
            "type": "function"
        }
        
        # Bind tool to the model
        chat_with_tools = chat.bind_tools([tool])
        
        # Create a sample message
        messages = [HumanMessage(content="What's the weather in New York?")]
        
        # Test invoke method
        chat_with_tools.invoke(messages)
        
        # Check the JSON payload
        json_payload = mock_post.call_args[1]["json"]
        self.assertIn("tools", json_payload)
        self.assertEqual(len(json_payload["tools"]), 1)
        self.assertEqual(json_payload["tools"][0]["function"]["name"], "get_weather")
    
    @mock.patch("langchain_asi.chat_models.requests.post")
    def test_with_structured_output(self, mock_post):
        """Test structured output."""
        # Configure the mock with a JSON response
        mock_response = mock.MagicMock()
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "asi1-mini",
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 7,
                "total_tokens": 20
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"title": "The Matrix", "year": 1999, "genre": ["Science Fiction", "Action"], "review": "A groundbreaking film.", "rating": 10}'
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ]
        }
        mock_response.raise_for_status = mock.MagicMock()
        mock_post.return_value = mock_response
        
        # Create the ASI1ChatModel
        chat = ASI1ChatModel(
            asi1_api_key="fake-api-key",
            model_name="asi1-mini"
        )
        
        # Define a structured output schema
        class MovieReview(BaseModel):
            """Movie review with title, year, and review text."""
            
            title: str = Field(description="The title of the movie")
            year: int = Field(description="The year the movie was released")
            genre: List[str] = Field(description="The genres of the movie")
            review: str = Field(description="A brief review of the movie")
            rating: int = Field(description="Rating from 1-10, with 10 being the best")
        
        # Create a structured output model
        structured_model = chat.with_structured_output(MovieReview)
        
        # Test invoke method
        response = structured_model.invoke("Write a review for The Matrix.")
        
        # Check that the response is as expected
        self.assertEqual(response.title, "The Matrix")
        self.assertEqual(response.year, 1999)
        self.assertEqual(response.genre, ["Science Fiction", "Action"])
        self.assertEqual(response.review, "A groundbreaking film.")
        self.assertEqual(response.rating, 10)
        
        # Check that the API was called with the right parameters
        mock_post.assert_called_once()
        
        # Check the JSON payload includes function call
        json_payload = mock_post.call_args[1]["json"]
        self.assertIn("functions", json_payload)
        self.assertIn("function_call", json_payload)


if __name__ == "__main__":
    unittest.main()