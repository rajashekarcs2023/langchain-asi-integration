"""Simple example of using the ASI-to-OpenAI adapter."""
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the ASI model and OpenAI adapter
from langchain_asi import ChatASI
from langchain_asi.openai_adapter import OpenAICompatibleASI

# Load environment variables
load_dotenv()

# Get API key from environment
asi_api_key = os.environ.get("ASI_API_KEY")
if not asi_api_key:
    raise ValueError("ASI_API_KEY environment variable not set")

print(f"API Key: {asi_api_key[:3]}...{asi_api_key[-3:]}")

# Define a simple schema for structured output
class StockAnalysis(BaseModel):
    """Schema for stock analysis."""
    company: str = Field(description="The company being analyzed")
    recommendation: str = Field(description="Buy, Sell, or Hold recommendation")
    reasons: List[str] = Field(description="Reasons for the recommendation")
    risk_level: str = Field(description="Low, Medium, or High risk assessment")

def test_basic_chat():
    """Test basic chat functionality with the adapter."""
    print("\n===== Testing Basic Chat =====\n")
    
    # Initialize the ASI model
    llm = ChatASI(
        model_name="asi1-mini",
        temperature=0.7,
        max_tokens=500,
        asi_api_key=asi_api_key,
        streaming=False,
        request_timeout=60.0
    )
    
    # Wrap it with the OpenAI compatibility adapter
    openai_compatible_llm = OpenAICompatibleASI(llm)
    
    # Test a simple chat
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What are the three most important factors to consider when investing in tech stocks?")
    ]
    
    print("Sending request to ASI API...")
    response = openai_compatible_llm.invoke(messages)
    
    print(f"Response: {response.content}")
    print("\nBasic chat test completed successfully!")

def test_structured_output():
    """Test structured output with the adapter."""
    print("\n===== Testing Structured Output =====\n")
    
    # Initialize the ASI model
    llm = ChatASI(
        model_name="asi1-mini",
        temperature=0.7,
        max_tokens=500,
        asi_api_key=asi_api_key,
        streaming=False,
        request_timeout=60.0
    )
    
    # Wrap it with the OpenAI compatibility adapter
    openai_compatible_llm = OpenAICompatibleASI(llm)
    
    # Create a structured output model
    structured_llm = openai_compatible_llm.with_structured_output(StockAnalysis)
    
    # Test structured output
    messages = [
        SystemMessage(content="You are a financial analyst specializing in tech stocks."),
        HumanMessage(content="Analyze Apple stock considering recent supply chain issues.")
    ]
    
    print("Sending request to ASI API for structured output...")
    try:
        response = structured_llm.invoke(messages)
        
        # Check if response is a StockAnalysis object
        if isinstance(response, StockAnalysis):
            print(f"Structured Response: {response}")
            print(f"Company: {response.company}")
            print(f"Recommendation: {response.recommendation}")
            print(f"Reasons: {response.reasons}")
            print(f"Risk Level: {response.risk_level}")
        else:
            print(f"Response is not a StockAnalysis object: {type(response)}")
            print(f"Content: {response.content if hasattr(response, 'content') else response}")
        
        print("\nStructured output test completed successfully!")
    except Exception as e:
        print(f"Error in structured output test: {e}")
        import traceback
        traceback.print_exc()

def test_tool_calling():
    """Test tool calling with the adapter."""
    print("\n===== Testing Tool Calling =====\n")
    
    # Define a simple calculator tool
    def calculator(a: int, b: int, operation: str) -> int:
        """Perform a simple calculation."""
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    # Initialize the ASI model
    llm = ChatASI(
        model_name="asi1-mini",
        temperature=0.7,
        max_tokens=500,
        asi_api_key=asi_api_key,
        streaming=False,
        request_timeout=60.0
    )
    
    # Wrap it with the OpenAI compatibility adapter
    openai_compatible_llm = OpenAICompatibleASI(llm)
    
    # Bind the calculator tool
    llm_with_tools = openai_compatible_llm.bind_tools([calculator])
    
    # Test tool calling
    messages = [
        SystemMessage(content="You are a helpful assistant with access to a calculator."),
        HumanMessage(content="What is 123 multiplied by 456?")
    ]
    
    print("Sending request to ASI API for tool calling...")
    try:
        response = llm_with_tools.invoke(messages)
        
        print(f"Response: {response.content}")
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"Tool Calls: {response.tool_calls}")
        
        print("\nTool calling test completed successfully!")
    except Exception as e:
        print(f"Error in tool calling test: {e}")
        import traceback
        traceback.print_exc()

# Run all tests
if __name__ == "__main__":
    test_basic_chat()
    test_structured_output()
    test_tool_calling()
