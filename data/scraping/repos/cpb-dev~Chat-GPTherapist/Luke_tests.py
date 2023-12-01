import openai, os

openai.api_key = "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"

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]



def collect_data(_):
    inpvalue = input()
    prompt = inpvalue
    context
