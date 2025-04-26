import os
from langchain_asi.chat_models import ChatASI
from langchain_core.messages import HumanMessage, SystemMessage

# Set your API key in the environment variable (you can also pass it directly)
# os.environ["ASI_API_KEY"] = "your-api-key-here"

# Initialize the chat model - it will automatically select the correct API endpoint
# based on the model name (asi1-mini uses https://api.asi1.ai/v1)
chat = ChatASI(model_name="asi1-mini")

# Create a simple conversation
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Tell me about the benefits of using LangChain with ASI models.")
]

# Generate a response
response = chat.invoke(messages)
print(f"Response: {response.content}")

# Example with streaming
print("\nStreaming response:")
# Create a streaming-enabled chat model
streaming_chat = ChatASI(model_name="asi1-mini", streaming=True)
for chunk in streaming_chat.stream(messages):
    print(chunk.content, end="", flush=True)
print("\n")

# Example with tool binding
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    """Input for weather information."""
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    unit: str = Field(default="fahrenheit", description="The temperature unit to use. Celsius or Fahrenheit.", alias="unit")
    
    # Configure model to allow field name aliases
    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }

# Bind the tool to the chat model
weather_chat = chat.with_structured_output(WeatherInput)

# Ask about the weather
weather_query = [
    SystemMessage(content="You are a helpful weather assistant. Extract the location and unit from the user's query. Respond with a valid JSON object containing the location and unit fields."),
    HumanMessage(content="What's the weather like in New York? Give me the temperature in celsius.")
]

# Get structured output
result = weather_chat.invoke(weather_query)
print(f"Structured output: {result}")
