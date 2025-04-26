from langchain_asi import ChatASI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain.tools import DuckDuckGoSearchRun

# Initialize the model
llm = ChatASI(model_name='asi1-mini')

# Create a simple search tool
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="search",
        func=search.run,
        description="Useful for searching the web for information."
    )
]

# Create a simple agent
system_prompt = """You are a helpful assistant that can search the web for information.
When you need to find information, use the search tool."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Test the agent
try:
    result = agent_executor.invoke({"messages": [{"role": "user", "content": "What is the current price of Apple stock?"}]})
    print("Agent result:", result)
except Exception as e:
    print(f"Error: {str(e)}")
    
    # Try a more direct approach
    print("\nTrying direct tool calling...")
    response = llm.invoke(
        "What is the current price of Apple stock?",
        tools=[{
            "name": "search",
            "description": "Useful for searching the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }]
    )
    print("Direct tool calling response:", response)
