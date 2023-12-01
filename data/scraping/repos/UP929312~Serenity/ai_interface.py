import json

from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

# https://www.python-engineer.com/posts/langchain-crash-course/
# https://python.langchain.com/en/latest/modules/agents/agent_executors/examples/agent_vectorstore.html

# from langchain.agents import load_tools
from langchain.memory import ConversationBufferWindowMemory

with open("assets/files/prompts.json", "r", encoding="utf-8") as file:
    prompts = json.load(file)

with open("keys/openai_key.txt", "r", encoding="utf-8") as file:
    api_key = file.read().strip()

# ========================================================================================================
# Main prompt template
# fmt: off
AGENT_TEMPLATE = prompts["system"] + "\n" + \
prompts["initial"] + \
"""

{history}
Human: {human_input}
Agent: """
# fmt: on
prompt = PromptTemplate(input_variables=["history", "human_input"], template=AGENT_TEMPLATE)
# ========================================================================================================

class AgentInterface:
    """A class which handles the interface between the application and LLM/Agent"""

    def __init__(self) -> None:
        self.chatgpt_chain = LLMChain(
            llm=ChatOpenAI(temperature=0, openai_api_key=api_key, client="idk", model_name="gpt-3.5-turbo"),  # gpt-3.5-turbo-0301
            prompt=prompt,
            verbose=False,  # Prints stuff to chat
            memory=ConversationBufferWindowMemory(k=2),
        )
        # tools = load_tools(["wikipedia"], llm=self.chatgpt_chain)  # In the future

    def continue_chain(self, human_input: str) -> str:
        """Takes an input of text from the user and sends it to the LLMChain, returning the output as a string"""
        output = self.chatgpt_chain.predict(human_input=human_input)
        return output.strip()


if __name__ == "__main__":
    agent = AgentInterface()
    import time

    start_time = time.time()
    for i in range(1):
        agent.continue_chain("How are you?")
    print(f"Time taken: {time.time() - start_time}")
    """
    while True:
        human_input = input("Human: ")
        output = agent.continue_chain(human_input)
        print(output)
    """
