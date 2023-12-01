import os
import openai

from getpass import getpass
# cohere_api_key = getpass("Enter your Cohere API key: ")
llm_type = input(
"""Which LLM do you want to use? Here are some LLM we support:
1. Cohere
2. Openai (GPT-3.5 or 4)
3. ai21
4. JinaChat
Type the number of the LLM you want to use and keep you API key handy: """)

env_updated = input("Have you created and updated your API keys in the .env file? (y/n): ")

if env_updated.capitalize() == "Y":
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv()) # read local .env file

    openai.api_key  = os.environ['OPENAI_API_KEY']
    cohere_api_key = os.environ['COHERE_API_KEY']
    ai21_api_key = os.environ['AI21_API_KEY']
    jinachat_api_key = os.environ['JINACHAT_API_KEY']
else:
    print("Please update your API keys in the .env file and run the script again.")

from ChatterBox import ChatterBox
from langchain.embeddings import CohereEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import (
    Cohere,
    AI21
)
from langchain.chat_models import ChatOpenAI, JinaChat



def main():
    """
    Main function to run the legal chatbot
    """
    print("Welcome to the legal chatbot!")
    print("Please wait while we load the document and set up the LLM...")
    web_url = "https://www.gesetze-im-internet.de/englisch_aufenthg/englisch_aufenthg.html"
    chunk_size = 500
    chunk_overlap = 20
    embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key)
    # embeddings = OpenAIEmbeddings()
    
    legal_bot = ChatterBox()
    legal_bot.load_document(media_type='web', url=web_url)
    legal_bot.split_document(split_type='recursive',
                             chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap,
                             doc_separator=["table of contentsSection"])
    legal_bot.get_vector_store_db(persist_directory='docs/chroma/', embeddings=embeddings)
    
    query = input("Hi, How can I help you with residence law?\n")
    
    chat_history = []
    if llm_type == "1":
        chat_llm = Cohere(cohere_api_key=cohere_api_key, temperature=0)
    elif llm_type == "2":
        chat_llm = ChatOpenAI(temperature=0)
    elif llm_type == "3":
        chat_llm = AI21(ai21_api_key=ai21_api_key)
    elif llm_type == "4":
        chat_llm = JinaChat(jinachat_api_key=jinachat_api_key)
    # chat_llm = Cohere(cohere_api_key=cohere_api_key, temperature=0)
    # chat_llm = OpenAI(temperature=0)
    # chat_llm = AI21(ai21_api_key=ai21_api_key)
    
    while query != 'exit':
        print("Thinking...")
        result, chat_history = legal_bot.get_answer_for_query_and_context_from_llm(
            query=query, 
            chat_history=chat_history, 
            llm=chat_llm)
        print(result["result"]) if result is not None else print("Sorry for the inconvenience, Please try again!")
        query = input("\nDo you have any other questions? If not, please type 'exit' to end the conversation.\n")

if __name__ == "__main__":
    main()