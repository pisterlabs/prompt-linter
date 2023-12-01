import openai
from config import key, location, endpoint, deployment_id_gpt4
openai.api_type = "azure"
openai.api_key = key
openai.api_base = endpoint

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

def generate_answer_from_context(user_input, context):
    conversation=[{"role": "system", "content": "Assistant is a large language model trained by OpenAI."}]
    prompt = create_prompt(context,user_input)            
    conversation.append({"role": "assistant", "content": prompt})
    conversation.append({"role": "user", "content": user_input})
    reply = generate_answer(conversation)
    return reply