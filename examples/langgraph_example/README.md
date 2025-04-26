# LangGraph with ChatASI Example

This example demonstrates how to use the ChatASI with LangGraph to create multi-agent systems that can collaborate to solve complex tasks.

## Overview

The examples in this directory showcase different ways to integrate ChatASI with LangGraph:

1. **agents_example.py**: Implements a research team with a supervisor, search agent, and SEC analyst agent that collaborate to answer financial queries.

2. **travel_assistant_example.py**: Creates a travel assistant that can search for hotels, restaurants, and attractions to help plan a vacation.

3. **multi_agent_example.py**: Demonstrates a multi-agent system with a supervisor, search agent, and SEC analyst that can answer financial questions.

4. **research_team.py**: Implements a research team that can search for information and analyze it to answer complex queries.

All examples use the real ChatASI model for language processing, while using mock data for tools like search, financial analysis, hotel bookings, restaurant recommendations, and attraction searches. This approach allows you to run the examples without needing access to external APIs.

## Setup

1. Install the required dependencies:

```bash
pip install -e .
```

2. Set up your ASI API key as an environment variable:

```bash
export ASI_API_KEY="your-api-key"
```

> **Note:** The ChatASI class automatically selects the correct API endpoint based on the model name:
> - 'asi1-mini' → https://api.asi1.ai/v1
> - other models → https://api.asi.ai/v1 (default)
>
> You can still override this by setting the ASI_API_BASE environment variable if needed.

The examples use a simple initialization pattern similar to ChatOpenAI:

```python
from langchain_asi import ChatASI

# Initialize the chat model
chat = ChatASI(model_name="asi1-mini")
# This will use ASI_API_KEY from environment variables
# and automatically select the correct API endpoint based on the model name
```

You can also set these values in your Python code (not recommended for production):

```python
import os
os.environ["ASI_API_KEY"] = "your-api-key"
# No need to set ASI_API_BASE unless you want to override the automatic selection
```

## Running the Examples

To run the agents example:

```bash
python examples/langgraph_example/agents_example.py
```

To run the travel assistant example:

```bash
python examples/langgraph_example/travel_assistant_example.py
```

To run the multi-agent example:

```bash
python examples/langgraph_example/multi_agent_example.py
```

To run the research team example:

```bash
python examples/langgraph_example/research_team.py
```

## Key Concepts

### Agent Nodes

Each example defines different agent nodes that perform specific tasks:

- **Supervisor Node**: Coordinates the workflow and decides which agent should handle each part of the query
- **Search Node**: Looks up general information
- **SEC Analyst Node**: Specializes in financial documents and analysis
- **Hotel Search Node**: Searches for hotels based on user criteria
- **Restaurant Search Node**: Searches for restaurants based on user criteria
- **Attraction Search Node**: Searches for attractions based on user criteria

### State Management

The examples use LangGraph's state management to track the conversation and pass information between agents. Each agent can update the state with its findings, which are then used by subsequent agents.

### Tool Mocking

Instead of making real API calls to external services, the examples use mock functions to simulate responses from tools like search engines, financial databases, hotel booking systems, etc. This allows the examples to run without requiring API keys or internet access.
