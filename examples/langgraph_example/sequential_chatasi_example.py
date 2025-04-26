"""Sequential example of using ChatASI without LangGraph."""
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_asi import ChatASI

# Set your API key - for a real implementation, use environment variables
# The ASI API key is loaded from the ASI_API_KEY environment variable
# You can set this with: export ASI_API_KEY="your-api-key"
os.environ["ASI_API_KEY"] = "sk_491aa37d22cc490883508f47e0c76e7abda7b212ab0642989937690bbd73a0b3"  # Replace with your actual API key

# Initialize the chat model
# The ASI API key should be set in your environment variables as ASI_API_KEY
chat = ChatASI(model_name="asi1-mini")

# Define the research function
def research_information(question: str) -> str:
    """Research information related to the question."""
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
    try:
        response = chain.invoke({"question": question})
        return response.content
    except Exception as e:
        return f"Error in research: {e}"

# Define the analysis function
def analyze_information(question: str, research: str) -> str:
    """Analyze the research information."""
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are an analyst. Analyze the following research information related to this question:
        
        Question: {question}
        
        Research Information:
        {research}
        
        Provide a detailed analysis of this information."""
    )
    
    # Create the chain
    chain = prompt | chat
    
    # Invoke the chain
    try:
        response = chain.invoke({"question": question, "research": research})
        return response.content
    except Exception as e:
        return f"Error in analysis: {e}"

# Define the summary function
def summarize_information(question: str, analysis: str) -> str:
    """Summarize the analysis into a concise response."""
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
    try:
        response = chain.invoke({"question": question, "analysis": analysis})
        return response.content
    except Exception as e:
        return f"Error in summary: {e}"

# Define the main function
def process_query(question: str) -> dict:
    """Process a user query using a sequential approach."""
    print(f"\nProcessing query: {question}\n")
    
    # Step 1: Research information
    print("Step 1: Researching information...")
    research = research_information(question)
    print(f"\nResearch Results:\n{research}\n")
    
    # Step 2: Analyze information
    print("Step 2: Analyzing information...")
    analysis = analyze_information(question, research)
    print(f"\nAnalysis:\n{analysis}\n")
    
    # Step 3: Summarize information
    print("Step 3: Summarizing information...")
    summary = summarize_information(question, analysis)
    print(f"\nSummary:\n{summary}\n")
    
    # Return the results
    return {
        "question": question,
        "research": research,
        "analysis": analysis,
        "summary": summary
    }

# Example usage
if __name__ == "__main__":
    # Example question
    question = "What are the financial prospects for Tesla in the next year?"
    
    # Process the query
    results = process_query(question)
    
    # Print the final summary
    print("\nFinal Answer:")
    print(results["summary"])
