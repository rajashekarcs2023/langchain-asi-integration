# LangChain ASI Integration

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Status-Beta-yellow" alt="Status">
</p>

## Overview

This package provides seamless integration between [LangChain](https://www.langchain.com/) and the [ASI AI](https://asi.ai) platform, allowing you to use ASI's language models as a drop-in replacement for OpenAI and other LLM providers in your LangChain applications.

**Key Features:**
- 🔄 **Direct OpenAI Replacement** - Use ASI models in place of OpenAI with minimal code changes
- 🛠️ **Tool Calling Support** - Bind tools to ASI models for function calling capabilities
- 📊 **Structured Output** - Generate validated data using Pydantic models
- 🔌 **OpenAI Compatibility Layer** - Use ASI with code designed for OpenAI
- 🧩 **LangGraph Integration** - Build complex agent workflows with ASI models

### Direct Replacement for ChatOpenAI

You can directly replace `ChatOpenAI` with `ChatASI` in your existing LangChain code:

```python
# Before: Using OpenAI with LangChain
from langchain_openai import ChatOpenAI

openai_llm = ChatOpenAI(api_key="your-openai-key")
response = openai_llm.invoke("What is the capital of France?")
```

```python
# After: Using ASI with LangChain (direct replacement)
from langchain_asi import ChatASI

asi_llm = ChatASI(asi_api_key="your-api-key")
response = asi_llm.invoke("What is the capital of France?")
```

### OpenAI Compatibility Layer (for LangGraph)

For more complex workflows that rely on OpenAI-specific features, especially with LangGraph, you can use the OpenAI compatibility adapter:

```python
from langchain_asi import ChatASI
from langchain_asi.openai_adapter import create_openai_compatible_model

# Create an ASI model
asi_model = ChatASI(asi_api_key="your-api-key")

# Make it OpenAI-compatible for existing code
openai_compatible = create_openai_compatible_model(asi_model)

# Use it with OpenAI-style interface
response = openai_compatible.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7
)
```

**Note:** While the OpenAI compatibility layer works for basic LangGraph workflows, complex LangGraph applications may require additional adjustments due to differences in how ASI and OpenAI handle tool calling and structured outputs. See the [Known Limitations](#known-limitations) section for details.

## 📋 Table of Contents

- [Installation](#installation)
- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
- [Working Features](#working-features)
- [Examples](#examples)
  - [Basic Examples](#basic-examples)
  - [Tool Calling](#tool-calling-1)
  - [Structured Output](#structured-output-1)
  - [OpenAI Compatibility](#openai-compatibility-1)
  - [LangGraph Integration](#langgraph-integration-1)
  - [Real-World Applications](#real-world-applications)
  - [Comparison Tests](#comparison-tests)
- [Usage Examples](#usage-examples)
- [Known Limitations](#known-limitations)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Quick Start

Since this package is not yet available on PyPI, you can install it directly from the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/your-username/langchain-asi-integration.git

# Navigate to the directory
cd langchain-asi-integration

# Install the package in development mode
pip install -e .

# Install required dependencies
pip install -r requirements.txt
```

### Setup Your API Key

Create a `.env` file in the root directory with your ASI API key:

```
ASI_API_KEY=your-api-key-here
```

Alternatively, set it as an environment variable:

```bash
export ASI_API_KEY="your-api-key-here"
```

### Verify Installation

Run this simple test to ensure everything is working:

```python
from langchain_asi import ChatASI
from langchain_core.messages import HumanMessage

# Initialize the model
chat = ChatASI()

# Test with a simple query
response = chat.invoke([HumanMessage(content="Hello, how are you?")])
print(response.content)
```

### Next Steps

Explore the [Examples](#examples) section below to see how to use various features of the integration.

## 📝 Requirements

- Python 3.8+
- An ASI API key (obtain from [ASI AI](https://asi1.ai))

### Dependencies

The following dependencies are required:

```
langchain-core>=0.1.0
httpx>=0.24.1
typing-extensions>=4.7.0
pydantic>=2.0.0
langchain>=0.3.0
```

For LangGraph integration, you'll also need:

```
langgraph>=0.0.26
```

## 🔑 Environment Setup

### Option 1: Environment Variable

Set your ASI API key as an environment variable:

```bash
export ASI_API_KEY="your-api-key"
```

### Option 2: .env File (Recommended for Development)

Create a `.env` file in your project directory:

```
ASI_API_KEY=your-api-key
```

### Option 3: Direct Initialization

Pass your API key directly when initializing the model:

```python
from langchain_asi import ChatASI

chat = ChatASI(asi_api_key="your-api-key")
```

## ✅ Working Features

The following features have been thoroughly tested and are working correctly:

### 1. Basic Chat Functionality
- Standard LangChain message-based interface
- Support for system, user, and assistant messages
- Temperature and max_tokens control
- Example: [Basic Chat Example](examples/basic_chat_example.py)

### 2. Tool Calling
- Support for binding tools to the model
- Automatic tool detection and execution
- Support for multiple tools in a single conversation
- Example: [Tool Calling Example](examples/comparison_tests/03_tool_calling.py)

### 3. Structured Output
- Generate validated structured data using Pydantic models
- Support for both function calling and JSON mode methods
- Example: [JSON Mode Example](examples/comparison_tests/04_json_mode.py)

### 4. OpenAI Compatibility Layer
- Use ASI models with code designed for OpenAI
- Support for OpenAI-style completions API
- Compatible with LangGraph and other OpenAI-based workflows
- Example: [OpenAI Adapter Example](examples/simple_adapter_example.py)

### 5. JSON Extraction
- Robust JSON extraction from various response formats
- Support for markdown code blocks and general text
- Example: [JSON Mode Example](examples/comparison_tests/04_json_mode.py)

## 📚 Examples

The repository contains a rich collection of examples demonstrating various features and use cases:

### Basic Examples

- [Basic Chat](examples/basic_chat_example.py) - Simple conversation with ASI
- [Advanced Chat](examples/advanced_chat_example.py) - More complex chat interactions
- [Chat Examples](examples/chat_examples.py) - Various chat patterns and techniques

### Tool Calling

- [Tool Calling](examples/comparison_tests/03_tool_calling.py) - Basic tool calling functionality
- [Advanced Tool Usage](examples/advanced_tool_usage.py) - Complex tool calling scenarios
- [Complex Tool Types](examples/complex_tool_types.py) - Working with complex tool schemas

### Structured Output

- [JSON Mode](examples/comparison_tests/04_json_mode.py) - Using JSON mode for structured output
- [Comprehensive Features Demo](examples/comprehensive_features_demo.py) - Demonstrating multiple features together

### OpenAI Compatibility

- [Simple Adapter Example](examples/simple_adapter_example.py) - Basic usage of the OpenAI adapter

### LangGraph Integration

- [Simple LangGraph Example](examples/simple_langgraph_example.py) - Basic LangGraph integration
- [Simple ChatASI Example](examples/langgraph_example/simple_chatasi_example.py) - Using ChatASI with LangGraph
- [Sequential ChatASI Example](examples/langgraph_example/sequential_chatasi_example.py) - Multi-step reasoning with LangGraph
- [Simple Graph](examples/langgraph_example/simple_graph.py) - Building graph structures with ChatASI
- [Multi-Agent Example](examples/langgraph_example/multi_agent_example.py) - Multiple agents working together
- [OpenAI Adapter Example](examples/langgraph_example/asi_openai_adapter_example.py) - Using the OpenAI adapter with LangGraph

### Real-World Applications

- [Financial Analysis](examples/financial_analysis_example.py) - Financial data analysis with ASI
- [Simple Financial Advisor](examples/simple_financial_advisor.py) - Building a financial advisor agent
- [Travel Assistant](examples/travel_assistant.py) - Creating a travel planning assistant
- [Trip Planner](examples/trip_planner.py) - Comprehensive trip planning system

### Comparison Tests

The `examples/comparison_tests/` directory contains examples that compare ASI with other LLM providers:

- [Basic Chat Comparison](examples/comparison_tests/01_basic_chat.py)
- [Streaming Comparison](examples/comparison_tests/02_streaming.py)
- [Tool Calling Comparison](examples/comparison_tests/03_tool_calling.py)
- [JSON Mode Comparison](examples/comparison_tests/04_json_mode.py)
- [Parallel Tool Calling](examples/comparison_tests/06_parallel_tool_calling.py)
- [Seed Parameter](examples/comparison_tests/07_seed_parameter.py)

## 📚 Usage Examples

Here are detailed examples of how to use the key features of the ASI-LangChain integration:

### Basic Chat

```python
import os
from langchain_asi import ChatASI
from langchain_core.messages import HumanMessage, SystemMessage

# Set your API key
os.environ["ASI_API_KEY"] = "your-api-key"

# Initialize the model
chat = ChatASI(model_name="asi1-mini")

# Basic usage
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]

response = chat.invoke(messages)
print(response.content)
```

### Tool Calling

```python
from langchain_core.tools import tool

@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the weather for a specific location."""
    # This would be a real API call in production
    return f"The weather in {location} is sunny and 25°{unit[0].upper()}"

# Bind the tool to the model
model_with_tools = chat.bind_tools([get_weather])

# Use the model with tools
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What's the weather like in Paris?")
]

response = model_with_tools.invoke(messages)
print(response.content)
```

### Structured Output

```python
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    rating: float = Field(description="Rating from 0-10")
    review: str = Field(description="Brief review of the movie")
    recommended: bool = Field(description="Whether you recommend this movie")

# Create a model with structured output
structured_model = chat.with_structured_output(MovieReview)

# Use the model to get structured output
messages = [
    SystemMessage(content="You are a movie critic."),
    HumanMessage(content="Review the movie 'The Matrix'")
]

response = structured_model.invoke(messages)
print(f"Title: {response.title}")
print(f"Rating: {response.rating}")
print(f"Review: {response.review}")
print(f"Recommended: {response.recommended}")
```

### JSON Mode

```python
# Create a model with structured output using JSON mode
json_model = chat.with_structured_output(
    MovieReview,
    method="json_mode"
)

# Use the model to get structured output in JSON format
messages = [
    SystemMessage(content="You are a movie critic."),
    HumanMessage(content="Review the movie 'Inception'")
]

response = json_model.invoke(messages)
print(f"Title: {response.title}")
print(f"Rating: {response.rating}")
print(f"Review: {response.review}")
print(f"Recommended: {response.recommended}")
```

### OpenAI Compatibility

```python
from langchain_asi import ChatASI
from langchain_asi.openai_adapter import create_openai_compatible_model

# Create an ASI model
asi_model = ChatASI(asi_api_key="your-api-key")

# Make it OpenAI-compatible
openai_compatible = create_openai_compatible_model(asi_model)

# Use it with OpenAI-style interface
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

response = openai_compatible.chat.completions.create(
    messages=messages,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### LangGraph Integration

For simple LangGraph workflows, it's recommended to use `ChatASI` directly rather than the OpenAI adapter:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_asi import ChatASI
from langgraph.graph import StateGraph, END

# Initialize the ChatASI model
chat = ChatASI(asi_api_key="your-api-key")

# Define a simple node function
def process_query(state):
    messages = state["messages"]
    response = chat.invoke(messages)
    return {"messages": messages + [response]}

# Create a simple graph
def create_simple_graph():
    # Define the graph
    builder = StateGraph()
    
    # Add the node
    builder.add_node("process", process_query)
    
    # Set the entry point
    builder.set_entry_point("process")
    
    # Set the exit point
    builder.add_edge("process", END)
    
    # Compile the graph
    return builder.compile()

# Use the graph
graph = create_simple_graph()
result = graph.invoke({"messages": [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]})

print(result["messages"][-1].content)
```

For more complex LangGraph workflows, be aware of the limitations described in the [Known Limitations](#known-limitations) section.

## ⚠️ Known Limitations

### 1. Tool Call Format
- ASI's response format differs from OpenAI's standard format for tool calls
- ASI provides tool calls in natural language rather than structured JSON
- The integration uses a compatibility layer to extract tool calls from natural language responses
- This extraction may not work perfectly for all response variations

### 2. Structured Output Format
- ASI does not natively return pure JSON in the same format as OpenAI
- The integration includes a robust JSON parser to extract structured data from various formats
- Complex nested structures may occasionally have parsing issues

### 3. LangGraph Compatibility
- **Direct ChatASI Integration**: Works well with LangGraph for both simple and sequential workflows
- **Sequential Workflows**: Multi-step reasoning chains with ChatASI work reliably
- **Simple Graph Structures**: Basic graph structures with single agents function correctly
- **Multi-Agent Systems**: Basic multi-agent workflows function but may have limitations with complex message passing
- **OpenAI Adapter**: The adapter initializes and runs with LangGraph but has limitations:
  - Agent interactions may not extract all information correctly
  - Complex message passing between agents may be incomplete
  - The adapter uses `bind_tools` instead of `bind` for tool binding
- **Recommendation**: Use direct ChatASI integration for LangGraph workflows when possible

### 4. Pydantic Compatibility
- The integration is transitioning from Pydantic v1 to v2
- Some deprecation warnings may appear when using certain features
- These warnings do not affect functionality but will be addressed in future updates

## 🔧 API Reference

### ChatASI

The main class for interacting with the ASI API.

```python
ChatASI(
    model_name: str = "asi1-mini",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    streaming: bool = False,
    asi_api_key: Optional[str] = None,
    asi_api_base: Optional[str] = None,
    verbose: bool = False,
    **kwargs
)
```

#### Parameters

- `model_name`: The name of the ASI model to use. Currently supports "asi1-mini".
- `temperature`: Controls randomness. Higher values make output more random, lower values make it more deterministic.
- `max_tokens`: Maximum number of tokens to generate in the response.
- `streaming`: Whether to stream the response or not.
- `asi_api_key`: Your ASI API key. If not provided, it will be read from the ASI_API_KEY environment variable.
- `asi_api_base`: The base URL for the ASI API. If not provided, it will be automatically determined based on the model name.
- `verbose`: Whether to print verbose output.

#### Methods

- `invoke(messages)`: Invoke the model with a list of messages.
- `bind_tools(tools, tool_choice=None)`: Bind tools to the model for function calling.
- `with_structured_output(schema, method="function_calling", include_raw=False)`: Create a model that returns structured output.
- `stream(messages)`: Stream the response from the model.
- `astream(messages)`: Asynchronously stream the response from the model.
- `ainvoke(messages)`: Asynchronously invoke the model.

### OpenAI Compatibility Adapter

```python
from langchain_asi.openai_adapter import create_openai_compatible_model

openai_compatible = create_openai_compatible_model(asi_model)
```

This adapter provides an OpenAI-compatible interface for ASI models, allowing them to be used with code designed for OpenAI, including LangGraph.

## 📚 Troubleshooting

### API Key Issues

If you encounter authentication errors, make sure your API key is correct and properly set:

```python
import os
os.environ["ASI_API_KEY"] = "your-api-key"

# Or pass it directly to the constructor
chat = ChatASI(asi_api_key="your-api-key")
```

### JSON Parsing Errors

If you encounter JSON parsing errors when using structured output, try using the function calling method instead of JSON mode:

```python
structured_model = chat.with_structured_output(
    YourSchema,
    method="function_calling"  # More reliable than "json_mode"
)
```

### Tool Call Detection Issues

If the model is not properly detecting or using tools, try making your prompts more explicit about using the available tools:

```python
messages = [
    SystemMessage(content="You are a helpful assistant. When asked about weather, use the get_weather tool."),
    HumanMessage(content="What's the weather like in Paris?")
]
```

### OpenAI Adapter Issues

If you encounter issues with the OpenAI adapter, make sure you're using the correct method names:

```python
# Use bind_tools instead of bind
model_with_tools = openai_compatible.bind_tools(tools=[weather_tool])
```

## 🤝 Contributing

Contributions to the ASI-LangChain integration are welcome! Here's how you can help:

1. **Report Issues**: If you find bugs or have feature requests, please open an issue on the GitHub repository.

2. **Submit Pull Requests**: Feel free to submit PRs for bug fixes or new features.

3. **Improve Documentation**: Help us improve the documentation by fixing errors or adding examples.

4. **Share Examples**: If you've created interesting examples using the integration, consider sharing them.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

<<<<<<< HEAD
- [LangChain](https://www.langchain.com/) for the amazing framework
- [ASI AI](https://asi.ai) for their powerful language models
- All contributors who have helped improve this integration
=======
<p align="center">
  Developed by <a href="https://github.com/rajashekarcs2023">Rajashekar Vennavelli</a>
</p>
>>>>>>> 32fcb92558e93631bfb2e6938768d11e97473c10
