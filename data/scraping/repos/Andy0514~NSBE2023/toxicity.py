import cohere
from cohere.classify import Example

co = cohere.Client('yiOWD4KfXSiayGiim2MRmZRUvGsbdEFOY5QaCQ1Z')

examples = [
  Example("you are hot trash", "Toxic"),  
  Example("go to hell", "Toxic"),
  Example("get rekt moron", "Toxic"),  
  Example("get a brain and use it", "Toxic"), 
  Example("say what you mean, you jerk.", "Toxic"), 
  Example("Are you really this stupid", "Toxic"), 
  Example("I will honestly kill you", "Toxic"),
  Example("You don't belong here", "Toxic"),
  Example("Do you not have a brain?", "Toxic"),
  Example("What is wrong with you?", "Toxic"),
  Example("Why are you so bad at this", "Toxic"),

  Example("yo how are you", "Benign"),  
  Example("I'm curious, how did that happen", "Benign"),  
  Example("Try that again", "Benign"),  
  Example("Hello everyone, excited to be here", "Benign"), 
  Example("I think I saw it first", "Benign"),  
  Example("That is an interesting point", "Benign"), 
  Example("I love this", "Benign"), 
  Example("We should try that sometime", "Benign"), 
  Example("You should go for it", "Benign"),
  Example("I know you can do this", "Benign"),
  Example("How are you feeling?", "Benign"),
  Example("You're an awesome person", "Benign"),
  Example("Thank you very much, you're my lifesaver", "Benign")
]

# inputs = [
#   "this game sucks, you suck",  
#     "stop being a dumbass",
#     "Let's do this once and for all",
#   "This is coming along nicely"  
# ]

def toxicity_filter(inputs):
  """
  Takes in a list of strings (sentences)
  Returns a list of strings and its toxicity
  """
  outputs = []
  response = co.classify(
      model='large',
      inputs=inputs,
      examples=examples)

  for i in range(len(inputs)):
    classification = response.classifications[i]
    if classification.prediction == "Toxic" and classification.confidence > 0.8:
      outputs.append([inputs[i], "toxic"])
    else:
      outputs.append([inputs[i], "benign"])
  return outputs


