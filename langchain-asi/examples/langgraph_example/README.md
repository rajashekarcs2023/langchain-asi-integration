# LangGraph with ASI1ChatModel Example

This example demonstrates how to use the ASI1ChatModel with LangGraph to create a research team of agents that can collaborate to answer financial queries.

## Overview

The example implements a research team with the following components:

1. **Search Agent**: Looks up general information on the internet
2. **SEC Analyst Agent**: Specializes in financial documents and SEC filings
3. **Supervisor Agent**: Coordinates the team and decides which agent should handle each part of the query

The agents are connected in a graph using LangGraph, allowing them to work together to answer complex financial queries.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set your ASI1 API key as an environment variable:

```bash
export ASI1_API_KEY=your_api_key_here
```

## Running the Example

To run the example, execute the following command:

```bash
python research_team.py
```

This will process a sample financial query about Tesla and display the conversation between the agents and the final result.

## Customization

You can modify the example to:

- Add more specialized agents
- Change the query processing logic
- Integrate with real search APIs and financial data sources
- Adjust the agent prompts and behavior

## How It Works

1. The user submits a financial query
2. The supervisor agent analyzes the query and decides which team member should handle it first
3. The selected agent processes the query and returns information
4. The supervisor reviews the information and decides the next step
5. This process continues until the supervisor determines that enough information has been gathered
6. The supervisor synthesizes a final response based on all the collected information
