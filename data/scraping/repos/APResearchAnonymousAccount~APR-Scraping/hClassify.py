import os
import openai
from dotenv import load_dotenv
import json
import time


load_dotenv()
with open('human.json') as f:
	humanPosts = json.load(f)
outfile = "ai3.json"
# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")
for post in humanPosts:
    start = time.time()
    prompt = "What is the political affiliation of someone who says: \""+post+"\""
    print(prompt)
    response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0.3, max_tokens=10)
    genText = str(response.choices[0].text)
    classification = ""
     
    
    print(genText)
    with open('human2.json','a') as f:
        f.write(","+json.dumps([post,genText]))
    time.sleep(time.time()-start+1.2)
    
print("Done")