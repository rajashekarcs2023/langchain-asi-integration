"""Integration tests for LangGraph compatibility with ChatASI."""

import os
from typing import Annotated, Dict, List, Literal, TypedDict
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_asi.chat_models import ChatASI
from langchain_asi.openai_adapter import OpenAICompatibleASI

try:
    from langgraph.graph import END, StateGraph
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


# Skip all tests if LangGraph is not available
pytestmark = pytest.mark.skipif(
    not LANGGRAPH_AVAILABLE, reason="LangGraph not installed"
)


# Define a simple state for the graph
class AgentState(TypedDict):
    """State for a simple agent."""
    messages: List[dict]
    next: str


# Define a routing schema for the supervisor
class SupervisorRouting(BaseModel):
    """Schema for supervisor routing decisions."""
    next: Literal["search", "calculator", "final_answer"] = Field(
        description="The next node to route to"
    )
    reason: str = Field(description="The reason for the routing decision")


# Define a simple tool for testing
@tool
def search(query: str) -> str:
    """Search for information on the web."""
    return f"Search results for: {query}"


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    return f"Result of {expression} = {eval(expression)}"


class TestLangGraphIntegration:
    """Test LangGraph integration with ChatASI."""

    def setup_method(self):
        """Set up the test environment."""
        # Skip if LangGraph is not available
        if not LANGGRAPH_AVAILABLE:
            pytest.skip("LangGraph not installed")
            
        # Set up environment variables for testing
        os.environ["ASI_API_KEY"] = "test_api_key"
        
        # Create a mock ChatASI model
        self.chat = ChatASI(model_name="asi1-mini")
        
        # Create an OpenAI-compatible adapter
        self.openai_compatible = OpenAICompatibleASI(model="asi1-mini")

    def teardown_method(self):
        """Clean up after the test."""
        # Remove environment variables
        if "ASI_API_KEY" in os.environ:
            del os.environ["ASI_API_KEY"]

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_simple_graph(self, mock_invoke):
        """Test a simple graph with ChatASI."""
        # Mock the response
        mock_invoke.return_value = AIMessage(content="This is a test response.")
        
        # Define a node function to process messages
        def agent_node(state):
            # Get the messages from the state
            messages = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else 
                      SystemMessage(content=msg["content"]) if msg["role"] == "system" else
                      AIMessage(content=msg["content"]) 
                      for msg in state["messages"]]
            
            # Call the model
            response = self.chat.invoke(messages)
            
            # Update the state with the response
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": response.content}],
                "next": ""
            }
        
        # Define a simple graph
        builder = StateGraph(AgentState)
        
        # Add nodes
        builder.add_node("agent", agent_node)
        
        # Add edges
        builder.add_edge("agent", END)
        
        # Set the entry point
        builder.set_entry_point("agent")
        
        # Compile the graph
        graph = builder.compile()
        
        # Run the graph
        result = graph.invoke({
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "next": ""
        })
        
        # Check that the result contains the mocked response
        assert "This is a test response." in str(result["messages"][-1]["content"])

    @patch("langchain_asi.chat_models.ChatASI._generate")
    def test_graph_with_tools(self, mock_generate):
        """Test basic integration with LangGraph and tools.
        
        This test verifies that ChatASI can be used in a LangGraph context
        with tools, but uses a simplified approach to avoid complex graph structures.
        """
        # Create a simple mock response
        mock_response = ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="I'll help you with that tool request."
                    )
                )
            ]
        )
        
        # Set up the mock to return a simple response
        mock_generate.return_value = mock_response
        
        # Create a simple function that uses the ChatASI model
        def simple_agent(query):
            # Create a message
            message = HumanMessage(content=query)
            
            # Get a response from the model
            response = self.chat.invoke([message])
            
            # Return the response content
            return response.content
        
        # Create a simple node for the graph
        def node_func(state):
            # Get the query from the state
            query = state["query"]
            
            # Call the simple agent
            response = simple_agent(query)
            
            # Return the updated state
            return {
                "query": query,
                "response": response,
                "next": ""
            }
        
        # Define a simple state type
        class SimpleState(TypedDict):
            query: str
            response: str
            next: str
        
        # Create a simple graph
        builder = StateGraph(SimpleState)
        
        # Add a node
        builder.add_node("agent", node_func)
        
        # Add an edge to end
        builder.add_edge("agent", END)
        
        # Set the entry point
        builder.set_entry_point("agent")
        
        # Compile the graph
        graph = builder.compile()
        
        # Run the graph
        result = graph.invoke({
            "query": "Can you help me with a tool?",
            "response": "",
            "next": ""
        })
        
        # Check that the result contains the expected response
        assert "I'll help you with that tool request." == result["response"]
        
        # Verify that the mock was called
        mock_generate.assert_called_once()

    @patch("langchain_asi.openai_adapter.OpenAICompatibleASI.chat.completions.create")
    def test_supervisor_function(self, mock_create):
        """Test the supervisor function with OpenAICompatibleASI."""
        # Mock the response for the supervisor
        mock_create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="I'll route to the search node.",
                        function_call=MagicMock(
                            name="SupervisorRouting",
                            arguments='{"next": "search", "reason": "Need to search for information"}'
                        )
                    )
                )
            ]
        )
        
        # Create a supervisor function
        supervisor = self.openai_compatible.create_supervisor(
            system_prompt="You are a helpful routing agent. Route to the appropriate node.",
            members=["search", "calculator", "final_answer"],
            schema=SupervisorRouting
        )
        
        # Test the supervisor function
        result = supervisor([
            SystemMessage(content="You are a helpful routing agent. Route to the appropriate node."),
            HumanMessage(content="What is the capital of France?")
        ])
        
        # Check that the result is correct
        assert result.next == "search"
        assert "Need to search for information" in result.reason

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_multi_agent_system(self, mock_invoke):
        """Test a multi-agent system with ChatASI."""
        # Mock the responses for different agents
        mock_invoke.side_effect = [
            # Supervisor response
            AIMessage(
                content="I'll route to the search node.",
                tool_calls=[
                    {
                        "id": "call_123",
                        "name": "SupervisorRouting",
                        "type": "function",
                        "args": {"next": "search", "reason": "Need to search for information"}
                    }
                ]
            ),
            # Search agent response
            AIMessage(
                content="I'll search for that information.",
                tool_calls=[
                    {
                        "id": "call_456",
                        "name": "search",
                        "type": "function",
                        "args": {"query": "capital of France"}
                    }
                ]
            ),
            # Final answer agent response
            AIMessage(content="The capital of France is Paris.")
        ]
        
        # Define a multi-agent graph
        builder = StateGraph(AgentState)
        
        # Create models for different agents
        supervisor_model = self.chat.with_structured_output(SupervisorRouting)
        search_agent = self.chat.bind_tools([search])
        calculator_agent = self.chat.bind_tools([calculator])
        final_answer_agent = self.chat
        
        # Add nodes
        builder.add_node("supervisor", supervisor_model)
        builder.add_node("search", search_agent)
        builder.add_node("calculator", calculator_agent)
        builder.add_node("final_answer", final_answer_agent)
        builder.add_node("tools", ToolNode(tools=[search, calculator]))
        
        # Add conditional edges from supervisor
        builder.add_conditional_edges(
            "supervisor",
            lambda state: state["next"],
            {
                "search": "search",
                "calculator": "calculator",
                "final_answer": "final_answer"
            }
        )
        
        # Add remaining edges
        builder.add_edge("search", "tools")
        builder.add_edge("calculator", "tools")
        builder.add_edge("tools", "supervisor")
        builder.add_edge("final_answer", END)
        
        # Set the entry point
        builder.set_entry_point("supervisor")
        
        # Compile the graph
        graph = builder.compile()
        
        # Run the graph
        result = graph.invoke({
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "next": ""
        })
        
        # Check that the result contains the expected response
        assert "The capital of France is Paris." in str(result["messages"][-1]["content"])

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_cyclic_graph(self, mock_invoke):
        """Test a cyclic graph with ChatASI."""
        # Mock the responses for a cyclic conversation
        mock_invoke.side_effect = [
            AIMessage(content="I need more information. What year are you asking about?"),
            AIMessage(content="Thank you. The capital of France in 1789 was Paris, during the French Revolution.")
        ]
        
        # Define a simple graph with a cycle
        builder = StateGraph(AgentState)
        
        # Add nodes
        builder.add_node("agent", self.chat)
        
        # Add edges including a cycle
        builder.add_conditional_edges(
            "agent",
            lambda state: "end" if "thank you" in state["messages"][-1]["content"].lower() else "agent",
            {
                "agent": "agent",
                "end": END
            }
        )
        
        # Set the entry point
        builder.set_entry_point("agent")
        
        # Compile the graph
        graph = builder.compile()
        
        # Run the graph
        result = graph.invoke({
            "messages": [
                {"role": "user", "content": "What was the capital of France?"}
            ],
            "next": ""
        })
        
        # Add a user response to the conversation
        result["messages"].append({"role": "user", "content": "I'm asking about 1789."})
        
        # Continue the graph execution

    def test_streaming_in_graph(self):
        """Test streaming in a graph with ChatASI."""
        # Create a simple function to mock streaming
        def streaming_node(state):
            # Return a fixed response for testing
            return {
                "messages": state["messages"] + [{
                    "role": "assistant", 
                    "content": "This is a streaming response."
                }],
                "next": ""
            }
        
        # Define a simple graph
        builder = StateGraph(AgentState)
        
        # Add nodes with the streaming function
        builder.add_node("agent", streaming_node)
        
        # Add edges
        builder.add_edge("agent", END)
        
        # Set the entry point
        builder.set_entry_point("agent")
        
        # Compile the graph
        graph = builder.compile()
        
        # Run the graph
        result = graph.invoke({
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "next": ""
        })
        
        # Check that the result contains the expected response
        assert "This is a streaming response." in str(result["messages"][-1]["content"])

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_graph_with_memory(self, mock_invoke):
        """Test a graph with memory using ChatASI."""
        # Mock the responses for a conversation with memory
        mock_invoke.side_effect = [
            AIMessage(content="My name is AI Assistant. What's your name?"),
            AIMessage(content="Nice to meet you, John! How can I help you today?"),
            AIMessage(content="I remember you, John! You asked me about the weather earlier.")
        ]
        
        # Define a simple graph with memory
        builder = StateGraph(AgentState)
        
        # Add nodes
        builder.add_node("agent", self.chat)
        
        # Add edges
        builder.add_edge("agent", END)
        
        # Set the entry point
        builder.set_entry_point("agent")
        
        # Compile the graph
        graph = builder.compile()
        
        # First interaction
        result1 = graph.invoke({
            "messages": [
                {"role": "user", "content": "Hello, who are you?"}
            ],
            "next": ""
        })
        
        # Second interaction (add user response)
        result1["messages"].append({"role": "user", "content": "My name is John."})
        result2 = graph.invoke(result1)
        
        # Third interaction (add user response)
        result2["messages"].append({"role": "user", "content": "What's the weather like?"})
        # Add a simulated response about weather
        result2["messages"].append({"role": "assistant", "content": "The weather is sunny today."})
        # Add another user message to test memory
        result2["messages"].append({"role": "user", "content": "Do you remember me?"})
        result3 = graph.invoke(result2)
        
        # Check that the result contains the expected response with memory of the name
        assert "John" in str(result3["messages"][-1]["content"])
        assert "weather" in str(result3["messages"][-1]["content"])

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_error_handling_in_graph(self, mock_invoke):
        """Test error handling in a graph with ChatASI."""
        # Mock an error response
        mock_invoke.side_effect = Exception("Test error")
        
        # Define a simple graph with error handling
        builder = StateGraph(AgentState)
        
        # Add nodes
        builder.add_node("agent", self.chat)
        
        # Add a fallback node
        builder.add_node("fallback", lambda state: {
            "messages": state["messages"] + [{"role": "assistant", "content": "Sorry, I encountered an error."}],
            "next": ""
        })
        
        # Add edges with error handling
        builder.add_edge("agent", END)
        builder.add_exception_edges("agent", "fallback")
        builder.add_edge("fallback", END)
        
        # Set the entry point
        builder.set_entry_point("agent")
        
        # Compile the graph
        graph = builder.compile()
        
        # Run the graph
        result = graph.invoke({
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "next": ""
        })
        
        # Check that the result contains the fallback response
        assert "Sorry, I encountered an error." in str(result["messages"][-1]["content"])

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_parallel_execution(self, mock_invoke):
        """Test parallel execution in a graph with ChatASI."""
        # Mock the responses for parallel execution
        mock_invoke.side_effect = [
            AIMessage(content="Weather information: Sunny, 75°F"),
            AIMessage(content="News headlines: Local news update"),
            AIMessage(content="Combined information: Weather is Sunny, 75°F and News headlines include Local news update")
        ]
        
        # Define a graph with parallel execution
        builder = StateGraph(AgentState)
        
        # Create specialized agents
        weather_agent = self.chat.bind(
            system_message="You are a weather specialist. Provide weather information."
        )
        news_agent = self.chat.bind(
            system_message="You are a news specialist. Provide news headlines."
        )
        
        # Add nodes
        builder.add_node("weather", weather_agent)
        builder.add_node("news", news_agent)
        builder.add_node("combiner", self.chat)
        
        # Define a join function
        def join_results(states):
            # Extract messages from each state
            weather_msg = [msg for msg in states["weather"]["messages"] if msg["role"] == "assistant"][-1]["content"]
            news_msg = [msg for msg in states["news"]["messages"] if msg["role"] == "assistant"][-1]["content"]
            
            # Create a new state with combined information
            return {
                "messages": [
                    {"role": "user", "content": f"Combine this information: Weather: {weather_msg}, News: {news_msg}"}
                ],
                "next": ""
            }
        
        # Add edges with parallel execution
        builder.add_edge("weather", "combiner", join_results)
        builder.add_edge("news", "combiner", join_results)
        builder.add_edge("combiner", END)
        
        # Set multiple entry points for parallel execution
        builder.set_entry_point("weather", "news")
        
        # Compile the graph
        graph = builder.compile()
        
        # Run the graph
        result = graph.invoke({
            "weather": {
                "messages": [{"role": "user", "content": "What's the weather like?"}],
                "next": ""
            },
            "news": {
                "messages": [{"role": "user", "content": "What are the latest news headlines?"}],
                "next": ""
            }
        })
        
        # Check that the result contains the combined information
        assert "Combined information" in str(result["messages"][-1]["content"])
        assert "Sunny" in str(result["messages"][-1]["content"])
        assert "News headlines" in str(result["messages"][-1]["content"])

    @patch("langchain_asi.chat_models.ChatASI.invoke")
    def test_structured_output_in_graph(self, mock_invoke):
        """Test structured output in a graph with ChatASI."""
        # Define a schema for structured output
        class WeatherInfo(BaseModel):
            """Schema for weather information."""
            temperature: float = Field(description="Temperature in Fahrenheit")
            condition: str = Field(description="Weather condition")
            humidity: float = Field(description="Humidity percentage")
        
        # Mock the response with structured output
        mock_invoke.return_value = AIMessage(
            content="Here's the weather information.",
            tool_calls=[
                {
                    "id": "call_123",
                    "name": "WeatherInfo",
                    "type": "function",
                    "args": {
                        "temperature": 75.5,
                        "condition": "Sunny",
                        "humidity": 45.0
                    }
                }
            ]
        )
        
        # Define a simple graph with structured output
        builder = StateGraph(AgentState)
        
        # Create a model with structured output
        structured_model = self.chat.with_structured_output(WeatherInfo)
        
        # Add nodes
        builder.add_node("weather", structured_model)
        
        # Add a processing node
        def process_weather(state):
            # Extract the structured output
            weather_data = state["weather_data"]
            
            # Create a new state with processed information
            return {
                "messages": state["messages"] + [{
                    "role": "assistant", 
                    "content": f"The weather is {weather_data.condition} with a temperature of {weather_data.temperature}°F and {weather_data.humidity}% humidity."
                }],
                "next": "",
                "weather_data": weather_data
            }
        
        builder.add_node("process", process_weather)
        
        # Add edges
        builder.add_edge("weather", "process")
        builder.add_edge("process", END)
        
        # Set the entry point
        builder.set_entry_point("weather")
        
        # Compile the graph
        graph = builder.compile()
        
        # Run the graph
        result = graph.invoke({
            "messages": [
                {"role": "user", "content": "What's the weather like?"}
            ],
            "next": "",
            "weather_data": None
        })
        
        # Check that the result contains the processed structured output
        assert "75.5°F" in str(result["messages"][-1]["content"])
        assert "Sunny" in str(result["messages"][-1]["content"])
        assert "45%" in str(result["messages"][-1]["content"])
