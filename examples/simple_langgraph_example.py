import os
import sys
from typing import Dict, List, Any, Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# Add the parent directory to the path so we can import our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LangGraph components
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Import ASI components
from langchain_asi import ChatASI
from langchain_asi.openai_adapter import create_openai_compatible_model

# Define the state schema for our graph
class AgentState(TypedDict):
    messages: List[Any]  # The conversation history
    next: str  # The next agent to call

# Define our tools/agents
def create_researcher_agent():
    """Create a researcher agent that looks up information."""
    # Create the ASI model
    asi_api_key = os.environ.get("ASI_API_KEY")
    if not asi_api_key:
        raise ValueError("ASI_API_KEY environment variable not set")
    
    # Initialize the model with proper parameters
    asi_model = ChatASI(
        model_name="asi1-mini",
        temperature=0.7,
        max_tokens=1000,
        asi_api_key=asi_api_key,
        streaming=False,
        request_timeout=60.0
    )
    
    # Create an OpenAI-compatible model
    openai_compatible_model = create_openai_compatible_model(asi_model)
    
    # Define the system prompt
    system_prompt = """
    You are a research assistant. Your task is to find information about the topic 
    the user is asking about. Be thorough and provide detailed information.
    """
    
    # Define the function to process the state
    def researcher_agent(state: AgentState) -> AgentState:
        # Get the messages from the state
        messages = state["messages"]
        
        # Add the system message
        full_messages = [SystemMessage(content=system_prompt)] + messages
        
        # Call the model
        response = openai_compatible_model.invoke(full_messages)
        
        # Add the response to the messages
        new_message = AIMessage(content=response)
        
        # Return the updated state
        return {"messages": messages + [new_message], "next": "writer"}
    
    return researcher_agent

def create_writer_agent():
    """Create a writer agent that summarizes information."""
    # Create the ASI model
    asi_api_key = os.environ.get("ASI_API_KEY")
    if not asi_api_key:
        raise ValueError("ASI_API_KEY environment variable not set")
    
    # Initialize the model with proper parameters
    asi_model = ChatASI(
        model_name="asi1-mini",
        temperature=0.7,
        max_tokens=1000,
        asi_api_key=asi_api_key,
        streaming=False,
        request_timeout=60.0
    )
    
    # Create an OpenAI-compatible model
    openai_compatible_model = create_openai_compatible_model(asi_model)
    
    # Define the system prompt
    system_prompt = """
    You are a professional writer. Your task is to take the research information 
    provided and create a well-structured, concise summary. Focus on clarity and brevity.
    """
    
    # Define the function to process the state
    def writer_agent(state: AgentState) -> AgentState:
        # Get the messages from the state
        messages = state["messages"]
        
        # Add the system message
        full_messages = [SystemMessage(content=system_prompt)] + messages
        
        # Call the model
        response = openai_compatible_model.invoke(full_messages)
        
        # Add the response to the messages
        new_message = AIMessage(content=response)
        
        # Return the updated state
        return {"messages": messages + [new_message], "next": END}
    
    return writer_agent

# Define the router function
def router(state: AgentState) -> str:
    """Route to the next agent based on the state."""
    return state["next"]

# Create the graph
def create_agent_graph():
    """Create a graph with a researcher and writer agent."""
    # Create the agents
    researcher = create_researcher_agent()
    writer = create_writer_agent()
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add the nodes
    workflow.add_node("researcher", researcher)
    workflow.add_node("writer", writer)
    
    # Add the edges
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)
    
    # Set the entry point
    workflow.set_entry_point("researcher")
    
    # Compile the graph
    return workflow.compile()

# Function to run the graph
def run_agent_graph(query: str):
    """Run the agent graph with the given query."""
    # Create the graph
    graph = create_agent_graph()
    
    # Create the initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next": "researcher"
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    # Return the final messages
    return result["messages"]

# Example usage
if __name__ == "__main__":
    # Test query
    query = "What are the main advantages and disadvantages of quantum computing?"
    
    print(f"\nQuery: {query}\n")
    
    # Run the graph
    try:
        messages = run_agent_graph(query)
        
        # Print the results
        print("\n=== Conversation ===\n")
        for i, message in enumerate(messages):
            if hasattr(message, 'content'):
                role = "User" if message.type == "human" else "AI"
                print(f"{role}: {message.content}\n")
    except Exception as e:
        print(f"Error: {e}")
