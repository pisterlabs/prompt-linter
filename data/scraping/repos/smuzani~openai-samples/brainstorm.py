import sys
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
chat = ChatOpenAI(temperature=0.69)
template="You are a helpful assistant with lots of creative ideas that helps users brainstorm ideas to a problem."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template= "{text}. Let's think step by step."
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)
if __name__ == "__main__":
    # Assuming the prompt is the second argument when running the script in zsh
    # Example usage: python script.py "What are some creative ways to save money?"
    text = sys.argv[1]
    response = chain.run(text=text)
    print(response)