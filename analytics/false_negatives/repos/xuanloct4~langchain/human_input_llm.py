import environment

from langchain.llms.human import HumanInputLLM
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

tools = load_tools(["wikipedia"])
# tools = load_tools(["arxiv"])
llm = HumanInputLLM(prompt_func=lambda prompt: print(f"\n===PROMPT====\n{prompt}\n=====END OF PROMPT======"))


# from langchain import PromptTemplate, LLMChain
# from langchain.llms import GPT4All
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# template = """Question: {question}
# Answer: Let's think step by step."""
# prompt = PromptTemplate(template=template, input_variables=["question"])
# local_path = '../gpt4all/chat/ggml-gpt4all-l13b-snoozy.bin'  # replace with your desired local file path
# # Callbacks support token-wise streaming
# callbacks = [StreamingStdOutCallbackHandler()]
# # Verbose is required to pass to the callback manager
# llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)


agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("What is Superstring Theory?")

# from langchain.utilities import WikipediaAPIWrapper
# wikipedia = WikipediaAPIWrapper()
# print(wikipedia.run('HUNTER X HUNTER'))