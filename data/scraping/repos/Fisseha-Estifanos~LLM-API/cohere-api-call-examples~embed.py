"""
An experiment script to understand the basics of working with the co:here API
Using the embed functionality
"""

import os
import sys
import cohere
sys.path.insert(1, 'scripts/')
from dotenv import load_dotenv

# load your environment
load_dotenv()
# use your own api key here
cohere_api_key = os.getenv('cohere_api_key')
co = cohere.Client(cohere_api_key)

response = co.embed(
  model='large',
  texts=["When are you open?", "When do you close?", "What are the hours?",
         "Are you open on weekends?", "Are you available on holidays?",
         "How much is a burger?", "What\'s the price of a meal?",
         "How much for a few burgers?", "Do you have a vegan option?",
         "Do you have vegetarian?", "Do you serve non-meat alternatives?",
         "Do you have milkshakes?", "Milkshake", "Do you have desert?",
         "Can I bring my child?", "Are you kid friendly?",
         "Do you have booster seats?", "Do you do delivery?",
         "Is there takeout?", "Do you deliver?", "Can I have it delivered?",
         "Can you bring it to me?", "Do you have space for a party?",
         "Can you accommodate large groups?", "Can I book a party here?"])

print('Embeddings: {}'.format(response.embeddings))
