from langchain.agents import AgentExecutor
from langchain.callbacks.base import BaseCallbackHandler

from src.agents.setup_agents import setup_agents
from src.prompt_processor.user_message_processor import UserMessageProcessor

prompt_processor = UserMessageProcessor()
agent_executor: AgentExecutor = setup_agents()


def generate_response(user_input):
    return agent_executor.run(input=user_input, callbacks=[BaseCallbackHandler()])


question = input("Type a message: ")
while question != "exit":
    output = generate_response(question)
    print(output)
    question = input("Type a message: ")
