import openai
from dotenv import dotenv_values

config = dotenv_values()
openai.api_key = config['OPENAI_KEY'] 

model_engine = "text-curie-001"
prompt = "Give me a fun fact about anything."

def respond(text_input):
    completion = openai.Completion.create(engine=model_engine,
                                          prompt=text_input,
                                          max_tokens=100,
                                          n=1,
                                          temperature=0.5,
                                          stop=None
                                          ) 
    output = completion.choices[0].text
    return output

def say_anything():
    return respond(prompt)