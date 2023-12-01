import openai
import spacy
from nltk.stem import WordNetLemmatizer
import random
import re

openai.api_key = "Your_API_Key_Goes_Here"

model_engine = "text-davinci-002"
model_prompts = {
    "Greeting": "Hello, I'm Jarvis. How can I assist you today?"
    #You Can Add More Greeting and what not
}

def generate_prompt(user_input):
    prompt_type = random.choice(list(model_prompts.keys()))
    prompt = model_prompts[prompt_type]
    return f"{prompt}\n{user_input}\n"

def generate_response(prompt, temperature=0.7):
    nlp = spacy.load("en_core_web_sm")
    lem = WordNetLemmatizer()
    doc = nlp(prompt)
    tokens = [lem.lemmatize(token.text) for token in doc]
    processed_prompt = " ".join(tokens)

    response = openai.Completion.create(
        engine=model_engine,
        prompt=processed_prompt,
        temperature=temperature,
        max_tokens=2048,
        n=3,
        stop="END\n",
        timeout=60,
    )
    message = response.choices[0].text
    message = re.sub(r"[^\x00-\x7f]+", "", message)
    message = message.strip()
    return message

prompt_shown = False
print(model_prompts["Greeting"]) 

while True:
    if not prompt_shown:
        prompt_type = random.choice(list(model_prompts.keys()))
    try:
        user_input = input("> ")
        prompt = generate_prompt(user_input)
        response = generate_response(prompt, temperature=0.7)
        print(response)
        prompt_shown = True
    except KeyboardInterrupt:
        break
