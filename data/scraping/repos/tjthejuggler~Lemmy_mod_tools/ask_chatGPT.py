import os
import openai

def send_request(prompt):
    request = [{"role":"user","content":prompt}]
    api_location = '~/projects/grammarpt/apikey.txt'
    api_location = os.path.expanduser(api_location)
    with open(api_location, 'r') as f:
        api_key = f.read().strip()
    openai.api_key = (api_key)    
    model_name = "gpt-3.5-turbo"    
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=request
    )
    response_string = response["choices"][0]["message"]["content"].replace("\n","").strip().lstrip()
 
    return response_string 

# title = "Inside the canine mind: A \"talking\" dog's owner on how to best connect with your furry pal'"
# print(send_request("Respond with a single type on animal and no punctuation. What animal might an article with the title '"+title+"' be about?"))