import cohere 
from cohere.classify import Example
import os
from dotenv import load_dotenv
load_dotenv()
cohere_api_key = os.environ['cohere_api_key']
print(cohere_api_key)
co = cohere.Client(cohere_api_key)

examples =  [
  Example("I feel like no one loves me", "Self-harm"),  
  Example("I feel meaningless", "Self-harm"),  
  Example("I want to feel pain", "Self-harm"),  
  Example("I want everything to end", "Self-harm"),  
  Example("Why does no one love me?", "Self-harm"),  
  Example("My chest hurts really badly. Please help!", "Medical attention"),  
  Example("My arm is broken", "Medical attention"),
  Example("I have a giant cut on my leg!", "Medical attention"),    
  Example("I feel like I'm going to pass out", "Medical attention"),
  Example("I think I'm getting warts on my genitals. What does that mean?", "Symptoms"),    
  Example("I have a slight fever and cough. What do I have?", "Symptoms"),    
  Example("I have diarrea and muscle aches. What do you think I have?", "Symptoms"),
  Example("I have a small headache and some trouble breathing. What does that mean?", "Symptoms")
]

inputs=[" God I hate life.",  
        "I feel short of breath and am feeling nauseous. What do I have?",  
        "I am bleeding out. Please help",
        "Hi, how are you?",
        ]
response = co.classify(  
    model='medium',  
    inputs=inputs,  
    examples=examples)

print(response.classifications)