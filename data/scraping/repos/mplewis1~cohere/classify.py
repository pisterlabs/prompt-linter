import cohere
import pandas as pd
import numpy as np
import altair as alt
import textwrap as tr

api_key = 'cUkUMhISEr8QsUhZ8uaVMxZtdL3UJrlaESCyNtHR'
co = cohere.Client(api_key)

from cohere.responses.classify import Example

examples = [Example("I'm so proud of you", "positive"), 
            Example("What a great time to be alive", "positive"), 
            Example("That's awesome work", "positive"), 
            Example("The service was amazing", "positive"), 
            Example("I love my family", "positive"), 
            Example("They don't care about me", "negative"), 
            Example("I hate this place", "negative"), 
            Example("The most ridiculous thing I've ever heard", "negative"), 
            Example("I am really frustrated", "negative"), 
            Example("This is so unfair", "negative"),
            Example("This made me think", "neutral"), 
            Example("The good old days", "neutral"), 
            Example("What's the difference", "neutral"), 
            Example("You can't ignore this", "neutral"), 
            Example("That's how I see it", "neutral")            
            ]

inputs=["Hello, world! What a beautiful day",
        "It was a great time with great people",
        "Great place to work",
        "That was a wonderful evening",
        "Maybe this is why",
        "Let's start again",
        "That's how I see it",
        "These are all facts",
        "This is the worst thing",
        "I cannot stand this any longer",
        "This is really annoying",
        "I am just plain fed up"
        ]

def classify_text(inputs, examples):

  response = co.classify(
    model='embed-english-v2.0',
    inputs=inputs,
    examples=examples)
  
  classifications = response.classifications
  
  return classifications

predictions = classify_text(inputs,examples)

classes = ["positive","negative","neutral"]
for inp,pred in zip(inputs,predictions):
  class_pred = pred.predictions[0]
  class_idx = classes.index(class_pred)
  class_conf = pred.confidences[0]

  print(f"Input: {inp}")
  print(f"Prediction: {class_pred}")
  print(f"Confidence: {class_conf:.2f}")
  print("-"*10)