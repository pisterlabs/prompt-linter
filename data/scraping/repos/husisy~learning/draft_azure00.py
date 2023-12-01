import os
import dotenv
import openai

dotenv.load_dotenv()
openai.api_key = os.environ['AZURE_API_KEY']
openai.api_base = os.environ['AZURE_ENDPOINT']
# example ENDPOINT: https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2023-05-15'
deployment_name = os.environ['AZURE_DEPLOYMENT_NAME']

# Send a completion call to generate an answer
start_phrase = 'Write a tagline for an ice cream shop. '
response = openai.Completion.create(engine=deployment_name, prompt=start_phrase, max_tokens=10)
text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
print(start_phrase+text)

response = openai.ChatCompletion.create(
    engine=deployment_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
        {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
        {"role": "user", "content": "Do other Azure Cognitive Services support this too?"}
    ]
)
# print(response)
print(response['choices'][0]['message']['content'])
