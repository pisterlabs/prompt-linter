import cohere  
import os 
from dotenv import load_dotenv
load_dotenv()



co = cohere.Client(os.environ['cohere_api'])
def scripter(Title, Year, Genre, plot):
    prompt =f"""Explain a movie named {Title}, {Genre}, {Year} of release,and its description is {plot}"""


    response = co.generate(  
        model='command-nightly',  
        prompt = prompt,  
        max_tokens=2000,  
        temperature=0.750)
    intro_paragraph = response.generations[0].text
    # print(intro_paragraph)
    return intro_paragraph
# print(intro_paragraph)

