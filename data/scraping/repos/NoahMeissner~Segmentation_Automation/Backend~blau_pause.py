import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_type = os.getenv('API_TYPE')
openai.api_base = os.getenv("API_BASE")
openai.api_version = os.getenv("API_VERSION")
openai.api_key = os.getenv("API_KEY")

s = """I want you to act as a Python interpreter. I will type commands and you will reply with what the
        python output should show. I want you to only reply with the terminal output inside one unique
        code block, and nothing else. Do no write explanations, output only what python outputs. Do not type commands unless I
        instruct you to do so. When I need to tell you something in English I will do so by putting
        text inside curly brackets like this: {example text}. My first command is a=1.
        the object is pencil, the color of the object is blue, 
        the material of the object is wood, and the size of the object is small. Output a Python tuple with the object as a String,
        the color as a BGR tuple, the material as a String, and the size of the object in meters as an int in that order. If the user's input
        for any of these parameters is unreasonable, label that slot with the bool False. Do not include any extra words in your answer. 
        If the user's input is not reasonable, write False."""

response = openai.ChatCompletion.create(
    engine="api3_1",
    messages=[{"role": "system", "content":s}],
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None)

print(response)
