from langchain_asi import ChatASI
from langchain_core.messages import HumanMessage
from agent.agents.supervisor import create_supervisor_agent
from agent.agents.search_agent import create_search_agent
from agent.agents.sec_agent import create_sec_agent
from agent.rag.chain import create_rag_chain
import os

# Initialize the model
llm = ChatASI(model_name='asi1-mini')

# Create agents
search_agent = create_search_agent(llm)
agent_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(agent_dir, "agent", "data", "raw", "apple_10k.pdf")
rag_chain = create_rag_chain(pdf_path)
sec_agent = create_sec_agent(llm, rag_chain)

# Create supervisor
supervisor = create_supervisor_agent(llm)

# Test the supervisor
print("Testing supervisor...")
query = "What are Apple's key financial risks and how have they changed over the past year?"
supervisor_result = supervisor.invoke({
    "messages": [HumanMessage(content=query)],
    "team_members": ["Search", "SECAnalyst"],
    "information_needed": [],
    "reasoning": ""
})
print("Supervisor result:", supervisor_result)

# Test the search agent
print("\nTesting search agent...")
search_result = search_agent.invoke({
    "messages": [HumanMessage(content=query)],
    "team_members": ["Search", "SECAnalyst"],
    "information_needed": [],
    "reasoning": ""
})
print("Search agent result:", search_result)

# Test the SEC agent
print("\nTesting SEC agent...")
sec_result = sec_agent.invoke({
    "messages": [HumanMessage(content=query)],
    "team_members": ["Search", "SECAnalyst"],
    "information_needed": [],
    "reasoning": ""
})
print("SEC agent result:", sec_result)
