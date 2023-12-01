import openai
import os  

########## AUTH & DEPLOYMENT

# setx GPT_CLASSIFICATOR_SECRET <API-KEY> => straight in VSC-Terminal for local deployment, not CMD
# print(GPT_CLASSIFICATOR_SECRET)

openai.api_key = os.environ.get('GPT_CLASSIFICATOR_SECRET')
openai.api_base = os.environ.get('GPT_CLASSIFICATOR_API_BASE')
openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future

deployment_name='YOURDEPLOYMENTHERE' #from azureopenai studio

########### GPT

json_names = "Alois, Amira, Arjun, Arvid, Bernhard, Cornelia, Elijah"

# Azure OpenAI call
completion = openai.ChatCompletion.create(
engine=deployment_name,
messages = [
    {"role": 'system', 'content': 'you are a web api. return result as valid json, grouped by classfication, respecting duplicates'},
    {'role': 'user', 'content': 'Classify each list in this name as female or male:' +json_names} 
],
temperature = 0.2
)

ai_response = completion['choices'][0]['message']['content']
print(ai_response)
