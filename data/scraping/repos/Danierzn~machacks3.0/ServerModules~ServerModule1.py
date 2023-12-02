import cohere
from cohere.classify import Example

co = cohere.Client('fZf3vVCtJkS69wYLEJWyr8WGRUupRJ4NnMSUwL0e') # API key

def rheaModel():
  """
  response = co.classify(  
    model='<model>',  
    inputs=inputs)
  """ 
  response = co.classify(
  model='48112639-5bee-4b80-8f70-1f5f60b645fe-ft',
  inputs=["I'm gonna kill myself.", "Today was really boring."]) # add more examples, here we should add parsed user data from social media accounts

  responses = [] 
  
  for line in response:
    responses.append(line.prediction) 

  return responses 
