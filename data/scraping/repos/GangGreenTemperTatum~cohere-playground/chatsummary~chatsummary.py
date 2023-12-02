import os
import re

import cohere 

# Create an `.env` file in the virtual project with line 'export CO_KEY="YOUR_API_KEY"' and import the os module
api_key = os.environ["CO_KEY"]
co = cohere.Client(api_key)

response = co.generate( 
  model='xlarge', 
  prompt="""Summarize this dialogue:
  Customer: Please connect me with a support agent.
  AI: Hi there, how can I assist you today?
  Customer: I forgot my password and lost access to the email affiliated to my account. Can you please help me?
  AI: Yes of course. First I\'ll need to confirm your identity and then I can connect you with one of our support agents.
  TLDR: A customer lost access to their account.
  --
  Summarize this dialogue:
  AI: Hi there, how can I assist you today?
  Customer: I want to book a product demo.
  AI: Sounds great. What country are you located in?
  Customer: I\'ll connect you with a support agent who can get something scheduled for you.
  TLDR: A customer wants to book a product demo.
  --
  Summarize this dialogue:
  AI: Hi there, how can I assist you today?
  Customer: I want to get more information about your pricing.
  AI: I can pull this for you, just a moment.
  TLDR:""", 
  max_tokens=20, 
  temperature=0.6, 
  k=0, 
  p=1, 
  frequency_penalty=0, 
  presence_penalty=0, 
  stop_sequences=["--"], 
) 
print('Prediction: {}'.format(response.generations[0].text)) 