import os
import cohere
from utils import message
from dotenv import load_dotenv

load_dotenv()
COHERE_API_KEY = f"{os.getenv('COHERE_API_KEY')}"
co_client = cohere.Client(COHERE_API_KEY)

def make_summary(text):
    try:
        response = co_client.summarize(
            text=text,
            model='summarize-xlarge',
            length='medium',
            extractiveness='medium',            
            format='paragraph',
        )    
        return response.summary
    except Exception as e :
        return message.message_error(500, e, "Internal Server Error")