"""Research team example using ChatASI."""
import os
from typing import Dict, List, TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_asi import ChatASI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# Set your API key - for a real implementation, use environment variables
# The ASI API key is loaded from the ASI_API_KEY environment variable
# You can set this with: export ASI_API_KEY="your-api-key"
os.environ["ASI_API_KEY"] = "sk_491aa37d22cc490883508f47e0c76e7abda7b212ab0642989937690bbd73a0b3"  # Replace with your actual API key

# Define the state
class ResearchState(TypedDict):
    """State for the research team graph."""
    question: str
    research: List[str]
    analysis: str
    summary: str

# Define the nodes
def researcher(state: ResearchState) -> ResearchState:
    """Research information related to the question."""
    # Initialize the chat model
    # The ASI API key should be set in your environment variables as ASI_API_KEY
    chat = ChatASI(model_name="asi1-mini")
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a researcher. Find information related to the following question:
        
        {question}
        
        Provide 3 key pieces of information that would help answer this question.
        Format your response as a numbered list."""
    )
    
    # Create the chain
    chain = prompt | chat
    
    # Invoke the chain
    response = chain.invoke({"question": state["question"]})
    
    # Extract the research points
    research_points = response.content.strip().split("\n")
    
    # Update the state
    return {"research": research_points}

def analyst(state: ResearchState) -> ResearchState:
    """Analyze the research information."""
    # Initialize the chat model
    # The ASI API key should be set in your environment variables as ASI_API_KEY
    chat = ChatASI(model_name="asi1-mini")
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are an analyst. Analyze the following research information related to this question:
        
        Question: {question}
        
        Research Information:
        {research}
        
        Provide a detailed analysis of this information."""
    )
    
    # Format the research information
    research_str = "\n".join([f"- {point}" for point in state["research"]])
    
    # Create the chain
    chain = prompt | chat
    
    # Invoke the chain
    response = chain.invoke({"question": state["question"], "research": research_str})
    
    # Update the state
    return {"analysis": response.content}

def summarizer(state: ResearchState) -> ResearchState:
    """Summarize the analysis into a concise response."""
    # Initialize the chat model
    # The ASI API key should be set in your environment variables as ASI_API_KEY
    chat = ChatASI(model_name="asi1-mini")
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a summarizer. Create a concise summary of the following analysis in response to the original question:
        
        Question: {question}
        
        Analysis: {analysis}
        
        Provide a clear, concise summary that directly answers the question."""
    )
    
    # Create the chain
    chain = prompt | chat
    
    # Invoke the chain
    response = chain.invoke({"question": state["question"], "analysis": state["analysis"]})
    
    # Update the state
    return {"summary": response.content}

# Build the graph
def build_graph():
    """Build the research team graph."""
    # Create the graph
    graph = StateGraph(ResearchState)
    
    # Add the nodes
    graph.add_node("researcher", researcher)
    graph.add_node("analyst", analyst)
    graph.add_node("summarizer", summarizer)
    
    # Add the edges
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "summarizer")
    graph.add_edge("summarizer", END)
    
    # Set the entry point
    graph.set_entry_point("researcher")
    
    # Compile the graph
    return graph.compile()

# Run the graph
def run_graph(question: str):
    """Run the research team graph."""
    # Build the graph
    graph = build_graph()
    
    # Run the graph
    result = graph.invoke({"question": question})
    
    # Return the result
    return result

# Example usage
if __name__ == "__main__":
    # Example question
    question = "What are the potential impacts of quantum computing on cybersecurity?"
    
    # Run the graph
    result = run_graph(question)
    
    # Print the result
    print("Research Points:")
    for i, point in enumerate(result["research"], 1):
        print(f"{i}. {point}")
    
    print("\nAnalysis:")
    print(result["analysis"])
    
    print("\nSummary:")
    print(result["summary"])
