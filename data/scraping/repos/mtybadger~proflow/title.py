import os
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv() 


llm = ChatOpenAI(temperature=0, model="gpt-4")

def generate_title(task):
    prompt_template = """
    You are an assistant developer chatbot. You are working on the below task:

    {task}

    Write a short, concise title for the potential pull request. Keep it under 10 words. Only return the title.
    """

    llm_chain = LLMChain.from_string(llm=llm, template=prompt_template)
    answer = llm_chain.predict(task=task)
    return answer
