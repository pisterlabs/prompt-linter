import os
import time
from langchain import Wikipedia
from langchain.agents import AgentType, Tool, initialize_agent, load_tools
from langchain.llms import LlamaCpp
from langchain.memory import ConversationSummaryBufferMemory


MODEL_PATH = "your/path/model.bin"
TEMP = float(os.environ.get("TEMP", 0.8))
NCTX = os.environ.get("NCTX", 8196)
os.environ["GOOGLE_API_KEY"] = "xxxx"
os.environ["GOOGLE_CSE_ID"] = "xxx"

llm = LlamaCpp(model_path=MODEL_PATH, temperature=TEMP, n_ctx=NCTX)
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=4098)

tools = load_tools(["google-search", "requests_all", "wikipedia", "human"], llm=llm)

print("Initializing agent...")
react = initialize_agent(tools, llm, memory=memory, n_batch=8, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True)
print("Agent initialized.")

goal = os.environ.get("AI_GOAL", "Where is Germany Located")
print(f"Goal: {goal}")

print("Executing agent with goal...")

start_time = time.time()

try:
    response = react({"input":goal})
except Exception as e:
    response = str(e)
    if response.startswith("Could not parse LLM output: `"):
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
        print("Corrected: "+response)

print(f"Response: {response}")

execution_time = time.time() - start_time
print(f'Execution time in seconds: {execution_time}')

# sanitized_output = response.replace("'", '"').replace("\n", " ").strip()
# print(response)
# os.system("say '" + sanitized_output + "'")
