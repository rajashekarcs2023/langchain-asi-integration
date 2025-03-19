"""Simple LangGraph example using ASI1ChatModel."""
import os
from typing import Annotated, Dict, List, TypedDict

import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
import functools

from langgraph.graph import StateGraph, END
from langchain_asi import ASI1ChatModel


# Set your API key - for a real implementation, use environment variables
os.environ["ASI1_API_KEY"] = "your-apikey"  # Replace with your actual API key


class ConversationState(TypedDict):
    """Define the state structure for the conversation."""
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


def search_node(state: Dict, llm: ASI1ChatModel) -> Dict:
    """Process a search request."""
    # Get the messages
    messages = state["messages"]
    
    # Add a system message
    system_message = SystemMessage(content="You are a search expert. Provide information about the query.")
    
    # Call the LLM
    response = llm.invoke([system_message] + messages)
    
    # Return the updated state
    return {"messages": state["messages"] + [response], "next": "analyst"}


def analyst_node(state: Dict, llm: ASI1ChatModel) -> Dict:
    """Process an analysis request."""
    # Get the messages
    messages = state["messages"]
    
    # Add a system message
    system_message = SystemMessage(content="You are a financial analyst. Analyze the information provided.")
    
    # Call the LLM
    response = llm.invoke([system_message] + messages)
    
    # Return the updated state
    return {"messages": state["messages"] + [response], "next": "FINISH"}


def create_graph():
    """Create a simple graph with two nodes."""
    # Initialize the LLM
    llm = ASI1ChatModel(
        model_name="asi1-mini",
        temperature=0.7,
    )
    
    # Create the nodes
    search = functools.partial(search_node, llm=llm)
    analyst = functools.partial(analyst_node, llm=llm)
    
    # Create the graph
    graph = StateGraph(ConversationState)
    
    # Add nodes
    graph.add_node("search", search)
    graph.add_node("analyst", analyst)
    
    # Add edges
    graph.add_conditional_edges(
        "search",
        lambda x: x["next"],
        {
            "analyst": "analyst",
            "FINISH": END
        },
    )
    
    graph.add_conditional_edges(
        "analyst",
        lambda x: x["next"],
        {
            "search": "search",
            "FINISH": END
        },
    )
    
    # Set entry point
    graph.set_entry_point("search")
    
    return graph.compile()


def main():
    """Run the example."""
    # Create the graph
    graph = create_graph()
    
    # Define the query
    query = "What are the recent financial performance and risk factors for Tesla?"
    
    # Run the graph
    result = graph.invoke({"messages": [HumanMessage(content=query)], "next": ""})
    
    # Print the result
    print("\nFinal Result:\n")
    for message in result["messages"]:
        print(f"{message.content}\n")


if __name__ == "__main__":
    main()
