import openai
from langchain.chat_models import ChatOpenAI
import os


openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {
            "role": "assistant",
            "content": "The Los Angeles Dodgers won the World Series in 2020.",
        },
        {"role": "user", "content": "Where was it played?"},
    ],
)

chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

clean_query = chat.call_as_llm("what are deez nuts?")


# %%
context = f"You are a large language model designed for blockchain and crypto-related activites. You have access to the following tools: {Tools}. "

initial_input = input("what would you like to do today?")

model_query = (
    context
    + "The user has made the following request: '"
    + initial_input
    + "'. What is the best tool for this purpose and what would the tool's input be? You MUST provide your answer in the form of a tuple (TOOL, INPUT), where TOOL is a tool you have access to, and INPUT is the text input to this tool."
)

result = chat.call_as_llm(model_query)

tool_options = tool_option_shower(result)

print(
    "Here are the most suitable tools and actions for this task - which one will you pick?"
)

print(tool_options)


# Variables:

# Input -> String
# Memory -> List[Facts()]
# Tools -> List[Tool()]


# Initialise:
# Memory ← []

# User presses submit:
# query ← USER_INPUT
# go to the lanchain NLP API


# # Agent needs to give a text description of the abstract plan it might cary out
# abstract_plan  ← agent_chat(memory, query)
# PRINT(abstract_plan)


# #
# best_tools <- agent_tool_selection(memory, query)[:3]
# List_tools_and_actions = [tool_action_generator(memiory, tool, query) for tool in best_tools]

# PRINT(list_tools_and_actions)


# a_tool_and_action_is_correct_for_taskl? <- USER_CLICK

# If a_tool_and_action_is_correct_for_taskl:

# # tool case
# Idx <- USER_CLICK
# Tool, action = List_tools_and_actions[idx]

# Fact = gen_fact(tool, action)
# Memory = update_memory(memory, fact)

# # agent suggests next tool to user:

# Else:
# 	Print(‘what went wrong and how can I do better?)
# 	Failure_fact = create_failure_fact()
# update_memory(failure_fact)

# %%
