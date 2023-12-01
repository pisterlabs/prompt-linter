import os
from random import choices 
import config
import openai 
openai.api_key = config.OPENAI_API_KEY 


def productDescription(query):
  response = openai.Completion.create(
  model="text-davinci-002",
  prompt="Generate a detailed product description for :{}".format(query),
  temperature=0.45,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0  
        )
  
  if 'choices' in response:
      if len(response['choices']) > 0 :
  
        answer = response['choices'][0]['text'] 
        
      else:
          answer = 'Opps sorry, you beat the AI this time'  

  else:
    answer = 'Opps sorry, you beat the AI this time'  

      
  return answer



def openAIQuery(query):
  response = openai.Completion.create(
  model="text-davinci-002",
  prompt=query,
  temperature=0.45,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0  
        )
  
  if 'choices' in response:
      if len(response['choices']) > 0 :
  
        answer = response['choices'][0]['text'] 
        
      else:
          answer = 'Opps sorry, you beat the AI this time'  

  else:
    answer = 'Opps sorry, you beat the AI this time'  

      
  return answer






