"""End-to-end tests for LangGraph integration with ChatASI."""

import os
from typing import Any, Dict, List, Optional, TypedDict, Annotated
import operator
import pytest

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from langchain_asi.chat_models import ChatASI
from langchain_asi.openai_adapter import create_openai_compatible_model


@pytest.mark.e2e
class TestLangGraphIntegration:
    """End-to-end tests for LangGraph integration with ChatASI."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Set up and tear down the test environment."""
        # Check if API key is available for integration tests
        if "ASI_API_KEY" not in os.environ:
            pytest.skip("ASI_API_KEY not found in environment variables")

        yield

    def test_simple_graph(self):
        """Test a simple LangGraph with ChatASI."""
        # Define the state
        class SimpleState(TypedDict):
            messages: List[BaseMessage]
            
        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")
        
        # Create a simple node
        def simple_node(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state["messages"]
            response = chat.invoke(messages)
            return {"messages": messages + [response]}
        
        # Build the graph
        graph = StateGraph(SimpleState)
        graph.add_node("simple", simple_node)
        graph.set_entry_point("simple")
        graph.add_edge("simple", END)
        
        # Compile the graph
        chain = graph.compile()
        
        # Run the graph
        messages = [HumanMessage(content="What is the capital of France?")]
        result = chain.invoke({"messages": messages})
        
        # Check the result
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][1], AIMessage)
        assert "paris" in result["messages"][1].content.lower()

    def test_multi_agent_graph(self):
        """Test a multi-agent LangGraph with ChatASI."""
        # Define the state
        class AgentState(TypedDict):
            messages: List[BaseMessage]
            next: str
            search_output: Optional[str]
            analyst_output: Optional[str]
            conversation_history: Annotated[List[Dict[str, str]], operator.add]
            
        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")
        
        # Define the supervisor node
        def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state["messages"]
            
            # Create the prompt template
            prompt = ChatPromptTemplate.from_template(
                """You are the supervisor of a research team. Your job is to:
                1. Understand the user's question
                2. Determine which team member should handle the query next
                3. Provide reasoning for your decision
                4. When all necessary information is gathered, synthesize a final response
                
                Available team members:
                - Search: Good for general information lookup
                - Analyst: Specialized in detailed analysis
                
                Question: {question}
                
                Current search information: {search_info}
                Current analysis information: {analysis_info}
                
                Respond with either "Search", "Analyst", or "FINISH" followed by your reasoning.
                """
            )
            
            # Get the last message
            last_message = messages[-1].content
            
            # Get the current search and analysis information
            search_info = state.get("search_output", "No information yet.")
            analysis_info = state.get("analyst_output", "No information yet.")
            
            # Invoke the chat model
            response = chat.invoke(
                prompt.format_messages(
                    question=last_message,
                    search_info=search_info,
                    analysis_info=analysis_info
                )
            )
            
            # Parse the response to determine the next step
            content = response.content.lower()
            if "search" in content[:10]:
                next_step = "Search"
            elif "analyst" in content[:10]:
                next_step = "Analyst"
            else:
                next_step = "FINISH"
                
            # Add to conversation history
            conversation_entry = {"role": "Supervisor", "content": response.content}
            
            # Return the updated state
            return {
                "messages": messages,
                "next": next_step,
                "conversation_history": [conversation_entry]
            }
            
        # Define the search node
        def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state["messages"]
            
            # Create the prompt template
            prompt = ChatPromptTemplate.from_template(
                """You are a search specialist. Provide general information about: {question}
                
                Keep your response concise and factual.
                """
            )
            
            # Get the last message
            last_message = messages[-1].content
            
            # Invoke the chat model
            response = chat.invoke(
                prompt.format_messages(question=last_message)
            )
            
            # Add to conversation history
            conversation_entry = {"role": "Search", "content": response.content}
            
            # Return the updated state
            return {
                "messages": messages,
                "search_output": response.content,
                "next": "",
                "conversation_history": [conversation_entry]
            }
            
        # Define the analyst node
        def analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state["messages"]
            search_output = state.get("search_output", "No search information available.")
            
            # Create the prompt template
            prompt = ChatPromptTemplate.from_template(
                """You are an analyst. Provide detailed analysis about: {question}
                
                Use this search information as a reference:
                {search_info}
                
                Provide a thorough analysis.
                """
            )
            
            # Get the last message
            last_message = messages[-1].content
            
            # Invoke the chat model
            response = chat.invoke(
                prompt.format_messages(
                    question=last_message,
                    search_info=search_output
                )
            )
            
            # Add to conversation history
            conversation_entry = {"role": "Analyst", "content": response.content}
            
            # Return the updated state
            return {
                "messages": messages,
                "analyst_output": response.content,
                "next": "",
                "conversation_history": [conversation_entry]
            }
            
        # Define the final node
        def final_node(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state["messages"]
            search_output = state.get("search_output", "No search information available.")
            analyst_output = state.get("analyst_output", "No analysis available.")
            
            # Create the prompt template
            prompt = ChatPromptTemplate.from_template(
                """You are the supervisor of a research team. Synthesize a final response to the user's question:
                
                Question: {question}
                
                Search information:
                {search_info}
                
                Analysis information:
                {analysis_info}
                
                Provide a comprehensive and helpful response.
                """
            )
            
            # Get the last message
            last_message = messages[-1].content
            
            # Invoke the chat model
            response = chat.invoke(
                prompt.format_messages(
                    question=last_message,
                    search_info=search_output,
                    analysis_info=analyst_output
                )
            )
            
            # Add to conversation history
            conversation_entry = {"role": "Final", "content": response.content}
            
            # Return the updated state
            return {
                "messages": messages + [response],
                "next": "",
                "conversation_history": [conversation_entry]
            }
            
        # Build the graph
        graph = StateGraph(AgentState)
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("search", search_node)
        graph.add_node("analyst", analyst_node)
        graph.add_node("final", final_node)
        
        # Add conditional edges
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state["next"],
            {
                "Search": "search",
                "Analyst": "analyst",
                "FINISH": "final"
            }
        )
        
        # Add edges back to supervisor
        graph.add_edge("search", "supervisor")
        graph.add_edge("analyst", "supervisor")
        
        # Add edge from final to END
        graph.add_edge("final", END)
        
        # Set the entry point
        graph.set_entry_point("supervisor")
        
        # Compile the graph
        chain = graph.compile()
        
        # Run the graph
        messages = [HumanMessage(content="What is the impact of quantum computing on cybersecurity?")]
        result = chain.invoke({
            "messages": messages,
            "next": "",
            "search_output": None,
            "analyst_output": None,
            "conversation_history": []
        })
        
        # Check the result
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][1], AIMessage)
        assert len(result["conversation_history"]) > 0
        
        # Check that the conversation history contains entries from different roles
        roles = [entry["role"] for entry in result["conversation_history"]]
        assert "Supervisor" in roles
        assert "Search" in roles or "Analyst" in roles
        assert "Final" in roles

    def test_openai_adapter_with_langgraph(self):
        """Test the OpenAI adapter with LangGraph."""
        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")
        
        # Create the OpenAI-compatible adapter
        adapter = create_openai_compatible_model(chat)
        
        # Define a simple function schema
        def create_supervisor(system_prompt, members):
            """Create a supervisor function for LangGraph."""
            # Create the prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])
            
            # Create the chain
            chain = prompt | adapter
            
            # Create the supervisor function
            def supervisor(state):
                # Get the input from the state
                input_text = state["input"]
                
                # Invoke the chain
                response = chain.invoke({"input": input_text})
                
                # Parse the response
                content = response.content.lower()
                
                # Determine the next step
                if any(member.lower() in content[:15] for member in members):
                    for member in members:
                        if member.lower() in content[:15]:
                            return {"next": member}
                else:
                    return {"next": "FINISH"}
            
            return supervisor
        
        # Create a supervisor
        supervisor = create_supervisor(
            system_prompt="You are a supervisor. Respond with either 'Agent1' or 'Agent2'.",
            members=["Agent1", "Agent2"]
        )
        
        # Define the state
        class SimpleState(TypedDict):
            input: str
            next: str
        
        # Build the graph
        graph = StateGraph(SimpleState)
        graph.add_node("supervisor", supervisor)
        
        # Add conditional edges
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state["next"],
            {
                "Agent1": END,
                "Agent2": END,
                "FINISH": END
            }
        )
        
        # Set the entry point
        graph.set_entry_point("supervisor")
        
        # Compile the graph
        chain = graph.compile()
        
        # Run the graph
        result = chain.invoke({
            "input": "Which agent should handle this task?",
            "next": ""
        })
        
        # Check the result
        assert "next" in result
        assert result["next"] in ["Agent1", "Agent2", "FINISH"]

    def test_tool_usage_in_langgraph(self):
        """Test tool usage in LangGraph with ChatASI."""
        # Define tools
        @tool
        def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return f"The weather in {location} is sunny."
        
        @tool
        def get_time(location: str) -> str:
            """Get the current time for a location."""
            return f"The current time in {location} is 12:00 PM."
        
        # Define the state
        class ToolState(TypedDict):
            messages: List[BaseMessage]
            tool_results: List[str]
        
        # Initialize the chat model
        chat = ChatASI(model_name="asi1-mini")
        
        # Bind tools to the model
        chat_with_tools = chat.bind_tools([get_weather, get_time])
        
        # Define the tool node
        def tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state["messages"]
            
            # Invoke the model with tools
            response = chat_with_tools.invoke(messages)
            
            # Check if there are tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_results = []
                
                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # Execute the tool
                    if tool_name == "get_weather":
                        result = get_weather(**tool_args)
                    elif tool_name == "get_time":
                        result = get_time(**tool_args)
                    else:
                        result = f"Unknown tool: {tool_name}"
                    
                    tool_results.append(result)
                
                # Return the updated state with tool results
                return {
                    "messages": messages + [response],
                    "tool_results": state.get("tool_results", []) + tool_results
                }
            else:
                # No tool calls, just return the response
                return {
                    "messages": messages + [response],
                    "tool_results": state.get("tool_results", [])
                }
        
        # Build the graph
        graph = StateGraph(ToolState)
        graph.add_node("tool", tool_node)
        graph.set_entry_point("tool")
        graph.add_edge("tool", END)
        
        # Compile the graph
        chain = graph.compile()
        
        # Run the graph
        messages = [
            SystemMessage(content="You are a helpful assistant that uses tools when appropriate."),
            HumanMessage(content="What's the weather in San Francisco?")
        ]
        result = chain.invoke({
            "messages": messages,
            "tool_results": []
        })
        
        # Check the result
        assert len(result["messages"]) > 1
        assert isinstance(result["messages"][-1], AIMessage)
        
        # Check if there are tool results
        if result["tool_results"]:
            assert any("weather in San Francisco" in result for result in result["tool_results"])
        else:
            # If no tool results, the model might have answered directly
            assert "weather" in result["messages"][-1].content.lower()
            assert "san francisco" in result["messages"][-1].content.lower()
