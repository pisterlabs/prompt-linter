import cohere 
from dotenv import load_dotenv
import os
import json

load_dotenv()


co = cohere.Client(os.environ["COHERE_API_KEY"])

def get_questions(question):
    response = co.chat( 
    model='command',
    message=f'give me some questions i can type into google search to further rabbithole into my initial question: \n\"{question}\"\nyou will return a response starting with \"some further questions you can ask are:\"\ndo not add any additional words after the questions are outputted',
    temperature=0.3,
    prompt_truncation='AUTO',
    stream=False,
    citation_quality='accurate',
    connectors=[{"id":"web-search"}],
    documents=[]
    ) 
    print(json.dumps(response.text))
    return response.text