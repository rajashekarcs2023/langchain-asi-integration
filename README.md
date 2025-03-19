# LangChain ASI Integration

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Status-Beta-orange" alt="Status">
</p>

## What is langchain-asi?

This package provides seamless integration between [LangChain](https://www.langchain.com/) and the [ASI AI](https://asi.ai) platform, allowing you to leverage ASI1's powerful language models with LangChain's extensive ecosystem of tools and abstractions.

With langchain-asi, you can:

- Use ASI1 models (like asi1-mini) as drop-in replacements for other LLMs in your LangChain applications
- Build complex chains and agents with ASI1 models
- Create structured outputs with validation
- Integrate with LangGraph for multi-agent workflows
- Stream responses for better user experience
- Use tool calling capabilities

## 📋 Table of Contents

- [Installation](#installation)
- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
- [Key Features](#key-features)
- [Usage Examples](#usage-examples)
  - [Basic Usage](#basic-usage)
  - [Tool Calling](#tool-calling)
  - [Structured Output](#structured-output)
  - [Streaming](#streaming)
  - [Financial Advisor Example](#financial-advisor-example)
  - [LangGraph Integration](#langgraph-integration)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### From Source (Current Method)

Clone the repository and install from source:

```bash
git clone https://github.com/rajashekarcs2023/langchain-asi-integration.git
cd langchain-asi-integration
pip install -e .
```

### From PyPI (Coming Soon)

Once published to PyPI, you'll be able to install the package using pip:

```bash
pip install langchain-asi  # Not available yet
```

## 📝 Requirements

- Python 3.8+
- An ASI1 API key (obtain from [ASI AI](https://asi.ai))

### Dependencies

The following dependencies will be automatically installed when you install the package:

```
langchain-core>=0.1.0
httpx>=0.24.1
requests>=2.31.0
typing-extensions>=4.7.0
pydantic>=2.0.0
python-dotenv>=1.0.0
langchain>=0.3.0
```

For LangGraph integration, you'll also need:

```
langgraph>=0.0.26
```

## 🔑 Environment Setup

### Option 1: Environment Variable

Set your ASI1 API key as an environment variable:

```bash
export ASI1_API_KEY="your-api-key"
```

### Option 2: .env File (Recommended for Development)

Create a `.env` file in your project directory:

```
ASI1_API_KEY=your-api-key
```

Then load it in your Python code:

```python
from dotenv import load_dotenv

load_dotenv()  # This will load the API key from .env
```

### Option 3: Direct Assignment (Not Recommended for Production)

```python
import os
os.environ["ASI1_API_KEY"] = "your-api-key"
```

## 🔄 Drop-in Replacement for Other LLMs

One of the key features of langchain-asi is that it allows you to use ASI1 models as drop-in replacements for other LLMs in your existing LangChain applications. Here's how you can switch from OpenAI to ASI1:

### Before (with OpenAI):

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key="your-openai-api-key"
)

response = llm.invoke("Tell me a joke about programming.")
print(response.content)
```

### After (with ASI1):

```python
from langchain_asi import ASI1ChatModel

llm = ASI1ChatModel(
    model_name="asi1-mini",
    temperature=0.7
)

response = llm.invoke("Tell me a joke about programming.")
print(response.content)
```

That's it! The ASI1ChatModel implements the same interface as other LangChain chat models, so you can use it in chains, agents, and other LangChain constructs without changing any other code.

## ✨ Key Features

- **ASI1ChatModel**: A LangChain chat model implementation for ASI1 API
  - Supports all standard LangChain chat model features
  - Fully compatible with LangChain's chains, agents, and memory systems
  - Configurable parameters like temperature, max_tokens, and top_p

- **Tool Calling**: Bind tools to the model for function calling capabilities
  - Define tools using Pydantic models
  - Automatic schema conversion and validation

- **Structured Output**: Generate and validate structured data
  - ASIJsonOutputParser for parsing JSON responses
  - ASIJsonOutputParserWithValidation for schema validation using Pydantic

- **Streaming**: Stream responses token by token for better user experience

- **LangGraph Integration**: Use ASI1 models in LangGraph workflows
  - Create multi-agent systems
  - Build complex decision trees and workflows

- **Memory Integration**: Works with LangChain's memory systems
  - ConversationBufferMemory
  - Other memory types

## 📚 Usage Examples

### Basic Usage

Here's a simple example of how to use the ASI1ChatModel for a basic chat interaction:

```python
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_asi import ASI1ChatModel

# Load API key from .env file
load_dotenv()

# Initialize the model
llm = ASI1ChatModel(
    model_name="asi1-mini",  # Use the ASI1 mini model
    temperature=0.7,       # Control randomness (0.0 to 1.0)
)

# Single message
response = llm.invoke("What is artificial intelligence?")
print(response.content)

# Multiple messages with system message
messages = [
    SystemMessage(content="You are a helpful AI assistant that specializes in explaining complex topics simply."),
    HumanMessage(content="Explain quantum computing to a 10-year-old.")
]

response = llm.invoke(messages)
print(response.content)
```

### Tool Calling

```python
from langchain_asi import ASI1ChatModel
from pydantic import BaseModel, Field
from typing import Optional

# Define a weather tool
class GetWeather(BaseModel):
    """Get the current weather in a given location"""
    
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: Optional[str] = Field(
        default="fahrenheit", 
        description="The unit of temperature, either 'celsius' or 'fahrenheit'"
    )

# Initialize the chat model
chat = ASI1ChatModel(
    model_name="asi1-mini",
    temperature=0.1,
)

# Bind the tool to the model
chat_with_tools = chat.bind_tools([GetWeather])

# Invoke the model with a question that requires the tool
response = chat_with_tools.invoke(
    "What's the weather like in Seattle?"
)

print(f"Content: {response.content}")
print(f"Tool calls: {response.tool_calls}")
```

### Structured Output

```python
from langchain_asi import ASI1ChatModel, ASIJsonOutputParserWithValidation
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Type

# Define a structured output schema
class MovieReview(BaseModel):
    """Movie review with title, year, and review text."""
    
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    genre: List[str] = Field(description="The genres of the movie")
    review: str = Field(description="A brief review of the movie")
    rating: int = Field(description="Rating from 1-10, with 10 being the best")

# Create a custom parser for MovieReview
class MovieReviewOutputParser(ASIJsonOutputParserWithValidation):
    pydantic_object: Type[BaseModel] = Field(default=MovieReview)

parser = MovieReviewOutputParser()

# Initialize the chat model
chat = ASI1ChatModel(
    model_name="asi1-mini",
    temperature=0.7,
)

# Create a chain with the parser
prompt_template = ChatPromptTemplate.from_template("{input}")
chain = prompt_template | chat | parser

# Generate a structured movie review
review = chain.invoke({
    "input": """Write a review for The Matrix. 
    
    Your response must be a valid JSON object with the following fields:
    - title: The title of the movie
    - year: The year the movie was released (as an integer)
    - genre: A list of genres for the movie
    - review: A brief review of the movie
    - rating: A rating from 1-10, with 10 being the best (as an integer)
    
    Format your entire response as a JSON object.
    """
})

print(f"Title: {review.title}")
print(f"Year: {review.year}")
print(f"Genres: {', '.join(review.genre)}")
print(f"Rating: {review.rating}/10")
print(f"Review: {review.review}")
```

### Streaming

```python
from langchain_asi import ASI1ChatModel
from langchain_core.messages import HumanMessage

chat = ASI1ChatModel(
    model_name="asi1-mini",
    temperature=0.7,
    streaming=True,
)

messages = [
    HumanMessage(content="Explain quantum computing in simple terms.")
]

for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)
```

### Financial Advisor Example

This example demonstrates how to build a simple financial advisor using ASI1ChatModel with multiple prompts and conversation memory:

```python
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_asi import ASI1ChatModel

# Load environment variables
load_dotenv()

# Initialize our ASI1 model
llm = ASI1ChatModel(
    model_name="asi1-mini",
    temperature=0.3,
)

# Create a memory for conversation context
memory = ConversationBufferMemory(return_messages=True)

# Create the risk assessment prompt
risk_prompt = ChatPromptTemplate.from_template("""Analyze the user's risk tolerance based on their financial question:
    
User question: {question}
    
Provide a risk assessment on a scale of 1-10, where:
1 = Extremely risk-averse
10 = Extremely risk-tolerant
    
Risk assessment:""")

# Create a simple financial advisor chain
def advisor_chain():
    # Step 1: Get risk assessment
    def get_risk_assessment(inputs):
        question = inputs["question"]
        risk_result = llm.invoke(risk_prompt.format(question=question))
        return {"risk_assessment": risk_result.content, "question": question}
    
    # Step 2: Get financial advice based on risk assessment
    # ... (additional chain steps)
    
    # Process the full chain
    def process_question(inputs):
        # Run all steps in sequence
        risk_result = get_risk_assessment(inputs)
        # ... (additional processing)
        return final_result
    
    return process_question

# Create and run the chain
chain = advisor_chain()
result = chain({"question": "Should I invest in high-growth tech stocks?"})
print(result["response"])
```

For the complete implementation, see the [examples directory](https://github.com/rajashekarcs2023/langchain-asi-integration/tree/main/examples).

### LangGraph Integration

LangChain ASI can be seamlessly integrated with LangGraph to create complex multi-agent workflows. Here's a simple example:

```python
import os
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_asi import ASI1ChatModel
from langgraph.graph import StateGraph, END

# Load API key from .env file
load_dotenv()

# Define state schema
class State(TypedDict):
    """Define the state structure for the conversation."""
    messages: Annotated[List[Dict], operator.add]
    next: str

def search_node(state: Dict, llm: ASI1ChatModel) -> Dict:
    """Process a search request."""
    messages = state["messages"]
    system_message = SystemMessage(content="You are a search expert. Provide information about the query.")
    response = llm.invoke([system_message] + messages)
    return {"messages": state["messages"] + [response], "next": "analyst"}

def analyst_node(state: Dict, llm: ASI1ChatModel) -> Dict:
    """Process an analysis request."""
    messages = state["messages"]
    system_message = SystemMessage(content="You are a financial analyst. Analyze the information provided.")
    response = llm.invoke([system_message] + messages)
    return {"messages": state["messages"] + [response], "next": "FINISH"}

# Build the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("search", search_node)
workflow.add_node("analyst", analyst_node)

# Add edges
workflow.add_edge("search", "analyst")
workflow.add_edge("analyst", END)

# Set the entry point
workflow.set_entry_point("search")

# Compile the graph
graph = workflow.compile()

# Run the graph
query = "What are the latest advancements in quantum computing?"
result = graph.invoke({"messages": [HumanMessage(content=query)], "next": ""})

# Print the results
for message in result["messages"]:
    if hasattr(message, "content"):
        print(f"{message.type}: {message.content}\n")
```

For more complex examples, check out the [LangGraph examples directory](examples/langgraph_example).

## 📖 API Reference

### ASI1ChatModel

```python
class ASI1ChatModel(BaseChatModel):
    """Chat model implementation for ASI1 API."""
    
    model_name: str = "asi1-mini"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    streaming: bool = False
    asi1_api_key: Optional[str] = None
    asi1_api_base: str = "https://api.asi.ai/v1"
    
    # ... methods and implementation details
```

### ASIJsonOutputParser

```python
class ASIJsonOutputParser(BaseOutputParser):
    """Output parser for handling JSON responses from ASI models."""
    
    # ... methods and implementation details
```

### ASIJsonOutputParserWithValidation

```python
class ASIJsonOutputParserWithValidation(ASIJsonOutputParser):
    """Output parser with Pydantic validation for JSON responses."""
    
    pydantic_object: Type[BaseModel]
    
    # ... methods and implementation details
```

## 🧪 Testing

The package includes a comprehensive test suite. To run the tests, first install the development dependencies:

```bash
pip install -e ".[dev]"
```

Or manually install the test dependencies:

```bash
pip install pytest pytest-cov
```

Then run the tests:

```bash
python -m pytest -v
```

The test suite includes tests for:

- Basic chat functionality
- Streaming
- Tool calling
- Output parsing with validation
- Memory integration
- LangGraph integration

### Test Coverage

To generate a test coverage report:

```bash
python -m pytest --cov=langchain_asi tests/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<p align="center">
  Developed by <a href="https://github.com/rajashekarcs2023">Rajashekar Vennavelli</a>
</p>
