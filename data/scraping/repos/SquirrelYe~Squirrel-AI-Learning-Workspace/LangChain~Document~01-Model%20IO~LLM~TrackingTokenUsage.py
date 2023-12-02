from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.callbacks import get_openai_callback

llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)

# 追踪单个token的使用情况
with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(cb)
    
# Tokens Used: 42
#         Prompt Tokens: 4
#         Completion Tokens: 38
# Successful Requests: 1
# Total Cost (USD): $0.00084

    
# 追踪多个token的使用情况
llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

with get_openai_callback() as cb:
    response = agent.run(
        "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?"
    )
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")
    
# > Entering new AgentExecutor chain...
#  I need to find out who Olivia Wilde's boyfriend is and then calculate his age raised to the 0.23 power.
# Action: Search
# Action Input: "Olivia Wilde boyfriend"
# Observation: Looks like Olivia Wilde and Jason Sudeikis are starting 2023 on good terms. Amid their highly publicized custody battle – and the actress' ...
# Thought: I need to find out Jason Sudeikis' age.
# Action: Search
# Action Input: "Jason Sudeikis age"
# Observation: 47 years
# Thought: I need to calculate 47 raised to the 0.23 power.
# Action: Calculator
# Action Input: 47^0.23
# Observation: Answer: 2.4242784855673896
# Thought: I now know the final answer.
# Final Answer: Jason Sudeikis, Olivia Wilde's boyfriend, is 47 years old and his age raised to the 0.23 power is 2.4242784855673896.

# > Finished chain.
# Total Tokens: 1536
# Prompt Tokens: 1368
# Completion Tokens: 168
# Total Cost (USD): $0.030719999999999997
# (.venv)