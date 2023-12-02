import openai
import os

openai.api_key = os.getenv("OPENAI_KEY")

def get_creator_bio(name, work):
    creator_data = {}
    creator_data["name"] = name

    prompt_text = (f"Generate a very short bio for {name}, creator of {work}. Don't unnecessarily emphasize {work} in the bio.")
    completion = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=prompt_text, max_tokens=3000, temperature=0.6, n=1)   
            
    raw_text = completion.choices[0].text.strip()
    creator_data["about"] = raw_text
     
    return creator_data