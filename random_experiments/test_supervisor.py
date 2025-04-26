from langchain_asi import ChatASI
from agent.utils.helpers import create_team_supervisor

# Initialize the model
llm = ChatASI(model_name='asi1-mini')

# Create the supervisor
members = ["Search", "SECAnalyst"]
system_prompt = "You are a team supervisor coordinating financial research."
supervisor = create_team_supervisor(llm, system_prompt, members)

# Test the supervisor with a simple query
result = supervisor.invoke({"messages": [{"role": "user", "content": "Who should analyze Apple's financial risks?"}]})
print("Result:", result)
