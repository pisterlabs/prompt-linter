import openai
key = ''
location = 'eastus'
endpoint = ''
openai.api_type = "azure"
openai.api_key = key
openai.api_base = endpoint
deployment_id_gpt4='gpt4'
openai.api_key = key

def create_prompt(context,query):
    header = "Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text and requires some latest information to be updated, print 'Sorry Not Sufficient context to answer query' \n"
    return header + context + "\n\n" + query + "\n"


def generate_answer(conversation):
    openai.api_version = "2023-03-15-preview"
    response = openai.ChatCompletion.create(
    engine=deployment_id_gpt4,
    messages=conversation,
    temperature=0,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop = [' END']
    )
    return (response['choices'][0]['message']['content']).strip()