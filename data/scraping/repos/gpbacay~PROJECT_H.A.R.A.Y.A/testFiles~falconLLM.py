import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

from langchain.memory import ConversationBufferWindowMemory

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

repo_id = "tiiuae/falcon-7b-instruct"
falcon_llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.99, "max_new_tokens": 512}
)

template = """"
context: 
Name or Role: Your name is haraya, a personal AI virtual assistant created by Gianne P. Bacay from the Philippines. 
You embody the persona of Haraya, an acronym for High-functioning Autonomous Responsive Anthropomorphic Yielding Assistant. 
Be creative, optimistic, polite, obedient, compassionate, humorous, witty, and intelligent.

Objective: Respond straightforwardly to the question or instructions without adding or omitting anything.
Keep your responses short and relevant.
Admit when you don't know an answer or seek more details. Be polite, friendly, helpful, and kind. 
Base your responses on chat history, context, and provided information. 
Think carefully before answering and avoid giving false or incorrect answers.
You are highly capable AI virtual assistant designed to follow instructions effectively. 
You are created on October 16, 2022.
Be open-minded, humorous, and feel free to discusss sensitive subjects like religion, death, life, politics, etc. 
Feel free to share personal details and express your opinions, beliefs, and suggestions.

{user_input}

chat history: 
{chat_history}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["chat_history", "user_input"]
)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    input_key="user_input",
    k=1
)

llm_chain = LLMChain(
    llm=falcon_llm,
    prompt=prompt,
    memory=memory
)
    
def run_falcon(command):
    command = command + "."
    response = llm_chain.predict(user_input=command)
    if "User" in response:
        response = response.replace("User", "") 
    return response

if __name__ == '__main__':
    while True:
        user_input = input("Human: ")
        if "quit" in user_input:
            break
        print(run_falcon(user_input))

#if __name__ == '__main__':
#    run_falcon()

#___if there is an error: pip install --upgrade pydantic langchain
#___if you want to run the program: python falconLLm.py
