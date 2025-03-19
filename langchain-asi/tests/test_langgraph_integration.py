"""Tests for the ASI1ChatModel integration with LangGraph."""
import os
import unittest
from unittest import mock

from langchain_asi import ASI1ChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph


class TestASI1ChatModelWithLangGraph(unittest.TestCase):
    """Test ASI1ChatModel integration with LangGraph."""

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
    def test_langgraph_integration(self, mock_post):
        """Test integration with LangGraph."""
        # Configure the mock
        mock_post.return_value = self.mock_response
        
        # Create the ASI1ChatModel
        chat = ASI1ChatModel(
            asi1_api_key="fake-api-key",
            model_name="asi1-mini"
        )
        
        # Define a simple node function
        def answer_question(state):
            """Answer the question."""
            question = state["question"]
            response = chat.invoke([HumanMessage(content=question)])
            state["answer"] = response.content
            return state
        
        # Create a simple graph
        workflow = StateGraph(input=dict, output=dict)
        workflow.add_node("answer", answer_question)
        workflow.set_entry_point("answer")
        workflow.set_finish_point("answer")
        
        # Compile the graph
        app = workflow.compile()
        
        # Test the graph
        result = app.invoke({"question": "Hello, how are you?"})
        
        # Check that the result is as expected
        self.assertEqual(result["answer"], "This is a test response.")
        
        # Check the JSON payload sent to the API
        json_payload = mock_post.call_args[1]["json"]
        self.assertEqual(json_payload["model"], "asi1-mini")
        self.assertEqual(json_payload["messages"][0]["role"], "user")
        self.assertEqual(json_payload["messages"][0]["content"], "Hello, how are you?")


if __name__ == "__main__":
    unittest.main()
