import os
from dotenv import load_dotenv
import cohere
from cohere.classify import Example

load_dotenv()
co = cohere.Client(os.getenv('COHERE_API_KEY','RMppgVMUjgiKZRWSIwjmmfbOwRa9YEhi1B15oxQ2'))

inputs=[
  "This item was broken when it arrived",
  "The product is amazing",
  "The product was not too bad",
]

examples=[
  Example("The order came 5 days early", "positive"), 
  Example("The item exceeded my expectations", "positive"), 
  Example("I ordered more for my friends", "positive"), 
  Example("I would buy this again", "positive"), 
  Example("I would recommend this to others", "positive"), 
  Example("The package was damaged", "negative"), 
  Example("The order is 5 days late", "negative"), 
  Example("The order was incorrect", "negative"), 
  Example("I want to return my item", "negative"), 
  Example("The item\'s material feels low quality", "negative"), 
  Example("The product was okay", "neutral"), 
  Example("I received five items in total", "neutral"), 
  Example("I bought it from the website", "neutral"), 
  Example("I used the product this morning", "neutral"), 
  Example("The product arrived yesterday", "neutral"),
]

res = co.classify(
  model='large',
  inputs=inputs,
  examples=examples,
)

print(res.classifications)