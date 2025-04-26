"""Multi-agent example using ChatASI with LangGraph."""
import os
from typing import Any, Dict, List, Optional, Sequence, TypedDict, Annotated, Literal, Union
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_asi import ChatASI
from langgraph.graph import StateGraph, END
import operator

# Set your API key - for a real implementation, use environment variables
os.environ["ASI_API_KEY"] = "sk_491aa37d22cc490883508f47e0c76e7abda7b212ab0642989937690bbd73a0b3"  # Replace with your actual API key

# Define the state type
class AgentState(TypedDict):
    """State for the agent graph."""
    messages: List[BaseMessage]
    next: str
    search_output: Optional[str]
    analyst_output: Optional[str]
    conversation_history: Annotated[List[Dict[str, str]], operator.add]

# Initialize the chat model
# The ASI API key should be set in your environment variables as ASI_API_KEY
chat = ChatASI(model_name="asi1-mini")

# Define the supervisor node
def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Supervisor node that coordinates the research team."""
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are the supervisor of a research team. Your job is to:
1. Understand the user's question
2. Determine which team member should handle the query next
3. Provide reasoning for your decision
4. When all necessary information is gathered, synthesize a final response

Available team members:
- Search: Good for general information lookup on the internet
- SECAnalyst: Specialized in financial documents and SEC filings

Question: {question}

Current conversation history:
{conversation_history}

Search output: {search_output}
Analyst output: {analyst_output}

Your response must be in the following format:

Thinking: Your step-by-step reasoning process
Next: [Search/SECAnalyst/FINISH]
Reasoning: Brief explanation of why you chose this next step"""
    )
    
    # Get the messages from the state
    messages = state["messages"]
    question = messages[0].content if messages else ""
    search_output = state.get("search_output", "Not available yet")
    analyst_output = state.get("analyst_output", "Not available yet")
    
    # Format the conversation history
    conversation_history = ""
    for entry in state.get("conversation_history", []):
        conversation_history += f"\n{entry['role']}: {entry['content']}\n"
    
    # Create the chain
    chain = prompt | chat
    
    # Invoke the chain
    response = chain.invoke({
        "question": question,
        "conversation_history": conversation_history,
        "search_output": search_output,
        "analyst_output": analyst_output
    })
    
    # Parse the response
    content = response.content
    
    # Extract the next step
    next_step = "FINISH"  # Default
    for line in content.split("\n"):
        if line.startswith("Next:"):
            next_step = line.replace("Next:", "").strip()
    
    # Add to conversation history
    conversation_entry = {"role": "Supervisor", "content": content}
    
    # If we have both outputs and this is at least the second iteration, default to FINISH
    if search_output != "Not available yet" and analyst_output != "Not available yet" and len(state.get("conversation_history", [])) >= 3:
        next_step = "FINISH"
    
    # Update the state
    return {
        "messages": messages + [response],
        "next": next_step,
        "conversation_history": [conversation_entry]
    }

# Define the search node
def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Search node that finds information on the internet."""
    # Get the last message
    messages = state["messages"]
    last_message = messages[0] if len(messages) == 1 else messages[-2]  # Get the original question
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a search expert who can find information on the internet.
        
        Please search for information related to: {query}
        
        Provide a detailed response with the most relevant information."""
    )
    
    # Create the chain
    chain = prompt | chat
    
    # Invoke the chain
    response = chain.invoke({"query": last_message.content})
    
    # Add to conversation history
    conversation_entry = {"role": "Search", "content": response.content}
    
    # Update the state
    return {
        "messages": messages + [response],
        "search_output": response.content,
        "conversation_history": [conversation_entry]
    }

# Define the SEC analyst node
def sec_analyst_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """SEC analyst node that analyzes financial documents."""
    # Get the last message
    messages = state["messages"]
    last_message = messages[0] if len(messages) == 1 else messages[-2]  # Get the original question
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a financial analyst specializing in SEC filings and financial documents.
        
        Please analyze the financial information related to: {query}
        
        Provide a detailed analysis based on the latest SEC filings and financial reports."""
    )
    
    # Create the chain
    chain = prompt | chat
    
    # Invoke the chain
    response = chain.invoke({"query": last_message.content})
    
    # Add to conversation history
    conversation_entry = {"role": "SECAnalyst", "content": response.content}
    
    # Update the state
    return {
        "messages": messages + [response],
        "analyst_output": response.content,
        "conversation_history": [conversation_entry]
    }

# Build the graph
def build_graph():
    """Build the agent graph."""
    # Create the graph
    graph = StateGraph(AgentState)
    
    # Add the nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search", search_node)
    graph.add_node("sec_analyst", sec_analyst_node)
    
    # Add the conditional edges
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "Search": "search",
            "SECAnalyst": "sec_analyst",
            "FINISH": END
        }
    )
    
    # Add the edges back to the supervisor
    graph.add_edge("search", "supervisor")
    graph.add_edge("sec_analyst", "supervisor")
    
    # Set the entry point
    graph.set_entry_point("supervisor")
    
    # Compile the graph
    return graph.compile()

# Run the graph
def run_graph(question: str):
    """Run the agent graph."""
    # Build the graph
    graph = build_graph()
    
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "next": "",
        "search_output": None,
        "analyst_output": None,
        "conversation_history": [{"role": "Human", "content": question}]
    }
    
    # Run the graph with a recursion limit
    try:
        result = graph.invoke(initial_state, {"recursion_limit": 10})
        return result
    except Exception as e:
        print(f"Error: {e}")
        # Return the partial state if available
        return initial_state

# Format the conversation
def format_conversation(state: Dict[str, Any]) -> str:
    """Format the conversation for display."""
    conversation = ""
    for entry in state.get("conversation_history", []):
        conversation += f"\n{entry['role']}: {entry['content']}\n"
    return conversation

# Example usage
if __name__ == "__main__":
    # Example question
    question = "What are the financial prospects for Tesla in the next year?"
    
    # Run the graph
    result = run_graph(question)
    
    # Print the conversation history
    print("\nConversation History:")
    print(format_conversation(result))
    
    # Print the final answer
    if result["next"] == "FINISH" and len(result["messages"]) > 1:
        print("\nFinal Answer:")
        print(result["messages"][-1].content)
