import cohere 
import os 
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('COHERE_API_KEY')
co = cohere.Client(api_key)

response = co.chat( 
  model='command',
  message="I'm a blind person without a cane. You are a camera that has detected several objects in my room and has to tell me what to do next. These are the approximations for where the objects in my room are based on what you, as the camera, can see: 1. Luggage at the bottom of the screen 2. Table on the left 3. Couch on the right. I am moving forward. Tell me what to do next",
  temperature=0.1,
  chat_history=[],
  prompt_truncation='auto',
  stream=False,
  citation_quality='accurate',
  connectors=[]
) 

print(response.message)
