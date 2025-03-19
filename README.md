# LangChain ASI Integration

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Status-Alpha-orange" alt="Status">
</p>

This package provides seamless integration between [LangChain](https://www.langchain.com/) and the ASI1 API, allowing you to leverage ASI1's powerful language models with LangChain's extensive ecosystem of tools and abstractions.

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

Install the package using pip:

```bash
pip install langchain-asi
```

## 📝 Requirements

- Python 3.8+
- An ASI1 API key (obtain from [ASI AI](https://asi.ai))
- LangChain Core (automatically installed as a dependency)

## 🔑 Environment Setup

Set your ASI1 API key as an environment variable:

```bash
export ASI1_API_KEY="your-api-key"
```

Or in your Python code (not recommended for production):

```python
import os
os.environ["ASI1_API_KEY"] = "your-api-key"
```

## ✨ Key Features

- **ASI1ChatModel**: A LangChain chat model implementation for ASI1 API
- **ASIJsonOutputParser**: Output parser for handling JSON responses from ASI models
- **ASIJsonOutputParserWithValidation**: Output parser with Pydantic validation for JSON responses
- **LangGraph Integration**: Support for using ASI1 models in LangGraph workflows
- **Tool Calling**: Support for function/tool calling with ASI1 models
- **Streaming**: Support for streaming responses from ASI1 models
- **Multi-step Chains**: Support for creating complex multi-step chains with ASI1 models
- **Conversation Memory**: Integration with LangChain's memory systems for stateful conversations

## 📚 Usage Examples

### Basic Usage

```python
from langchain_asi import ASI1ChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize the chat model
chat = ASI1ChatModel(
    model_name="asi1-mini",
    temperature=0.7,
)

# Create messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me a short joke about programming.")
]

# Generate response
response = chat.invoke(messages)
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
    max_tokens=4000
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

The complete implementation includes:

1. Risk assessment based on the user's question
2. Personalized financial advice generation based on the risk assessment
3. Conversational response formatting
4. Memory for maintaining conversation context

For the complete implementation, see the [examples directory](https://github.com/rajashekarcs2023/langchain-asi-integration/tree/main/langchain-asi/examples).

### LangGraph Integration

The package includes examples of integrating ASI1ChatModel with LangGraph for creating complex, multi-agent workflows.

#### Simple Graph Example

```python
import os
from typing import Annotated, Dict, List, TypedDict
import operator
import functools

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_asi import ASI1ChatModel

# Set your API key
os.environ["ASI1_API_KEY"] = "your-api-key"

class ConversationState(TypedDict):
    """Define the state structure for the conversation."""
    messages: Annotated[List[BaseMessage], operator.add]
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

def create_graph():
    """Create a simple graph with two nodes."""
    llm = ASI1ChatModel(model_name="asi1-mini", temperature=0.7)
    
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
    graph = create_graph()
    query = "What are the recent financial performance and risk factors for Tesla?"
    result = graph.invoke({"messages": [HumanMessage(content=query)], "next": ""})
    
    print("\nFinal Result:\n")
    for message in result["messages"]:
        print(f"{message.content}\n")

if __name__ == "__main__":
    main()
```

For more complex examples, see the [examples directory](https://github.com/rajashekarcs2023/langchain-asi-integration/tree/main/langchain-asi/examples).

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

The package includes comprehensive tests for all functionality. To run the tests:

```bash
pip install pytest
pytest
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
  Made with ❤️ by <a href="https://github.com/rajashekarcs2023">Rajashekar Vennavelli</a>
</p>
