from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import os
import dotenv

# load dot env
dotenv.load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")

prompt = """For the section of a blog given, return a comma-separated list of keywords from the text and topics matching the section. No preamble."""

def generate_completion(text, prompt_=prompt):

    llm = ChatOpenAI(
        openai_api_key=OPENAI_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0)
    
    ## Initialize messages for chat 
    messages = [SystemMessage(content=prompt_)]

    messages.append(HumanMessage(content=text))

    response = llm(messages)

    return response.content

def get_keyword(prompt):
    text = generate_completion(prompt)
    listofkeywords = text[:len(text)-1].split(',')
    listofkeywords = [keyword.strip().lower() for keyword in listofkeywords]
    return listofkeywords

# if __name__ == '__main__':

#     print(get_keyword('''heading:  What is SOC 2 compliance? 
#  content:  SOC 2 or formally Service Organization Control 2, is a security and privacy compliance standard that provides assurance to customers that a service provider has implemented appropriate controls to protect their data. The American Institute of CPAs (AICPA) produced SOC 2, a voluntary compliance standard for service organizations, which outlines how businesses should safeguard client data.
# Each organization's specific demands are taken into account while creating a SOC 2 report. Every organization has the ability to develop controls that adhere to one or more of SOC 2 trust principles depending on its unique business practices. These internal reports offer crucial details about how they handle their data to authorities, partners in business, and suppliers.'''))