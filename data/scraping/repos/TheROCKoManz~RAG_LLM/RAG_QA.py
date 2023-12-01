from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from utils import build_model
import os
from langchain.llms import OpenAI

from warnings import filterwarnings
filterwarnings('ignore')

def AskQuestion():
    qa_model = build_model()
    welcome_prompt = \
    '''
    Hi, I am a Retrieval Augmented Generative AI.
    Currently I am trained to answer questions 
    based on a Document regarding PAN card informations.

    Feel Free to ask me any questions and I'll try to answer...\n
    '''
    print(welcome_prompt)

    while(True):
        question = input("\nYour question: ")
        if len(question) == 0:
            continue
        print('\nAI-Says:', end='')
        print(qa_model({"query": question})['result'])

        nxt = input().lower()
        if len(nxt)>0 and nxt in ['quit', 'exit', 'close', 'stop']:
            break

    print('\n\nThank You for using this Document Retriever. Hope to see you soon again...!!\n\n')


if __name__ == "__main__":
    AskQuestion()
    os.environ['OPENAI_API_KEY'] = ''
