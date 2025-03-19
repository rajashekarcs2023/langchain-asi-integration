"""Tests for the ASI1ChatModel integration with memory."""
import os
import unittest
from unittest import mock

from langchain_asi import ASI1ChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory


class TestASI1ChatModelWithMemory(unittest.TestCase):
    """Test ASI1ChatModel integration with memory."""

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
    def test_memory_integration(self, mock_post):
        """Test integration with ConversationBufferMemory."""
        # Configure the mock
        mock_post.return_value = self.mock_response
        
        # Create the ASI1ChatModel
        chat = ASI1ChatModel(
            asi1_api_key="fake-api-key",
            model_name="asi1-mini"
        )
        
        # Create a memory instance
        memory = ConversationBufferMemory(return_messages=True)
        
        # Add a message to memory
        memory.save_context(
            {"input": "Hello, how are you?"}, 
            {"output": "I'm doing well, thank you for asking!"}
        )
        
        # Get the chat history from memory
        chat_history = memory.load_memory_variables({})
        
        # Verify the chat history
        self.assertIn("history", chat_history)
        self.assertEqual(len(chat_history["history"]), 2)
        
        # Create a new message
        new_message = HumanMessage(content="What's the weather like today?")
        
        # Combine chat history with new message
        messages = chat_history["history"] + [new_message]
        
        # Test invoke method
        response = chat.invoke(messages)
        
        # Check that the response is as expected
        self.assertEqual(response.content, "This is a test response.")
        
        # Save the new interaction to memory
        memory.save_context(
            {"input": new_message.content}, 
            {"output": response.content}
        )
        
        # Get the updated chat history
        updated_chat_history = memory.load_memory_variables({})
        
        # Verify the updated chat history
        self.assertIn("history", updated_chat_history)
        self.assertEqual(len(updated_chat_history["history"]), 4)
        
        # Check the JSON payload sent to the API
        json_payload = mock_post.call_args[1]["json"]
        self.assertEqual(json_payload["model"], "asi1-mini")
        self.assertEqual(len(json_payload["messages"]), 3)
        self.assertEqual(json_payload["messages"][0]["role"], "user")
        self.assertEqual(json_payload["messages"][0]["content"], "Hello, how are you?")
        self.assertEqual(json_payload["messages"][1]["role"], "assistant")
        self.assertEqual(json_payload["messages"][1]["content"], "I'm doing well, thank you for asking!")
        self.assertEqual(json_payload["messages"][2]["role"], "user")
        self.assertEqual(json_payload["messages"][2]["content"], "What's the weather like today?")


if __name__ == "__main__":
    unittest.main()
