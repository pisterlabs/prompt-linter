

def gpt_init(openai):
    import api_key

    #import openai
    openai_key = api_key.openai_key()
    openai.api_key = openai_key


def generate_postcard(openai,_keywords):
    #import os
    keywords = _keywords
    prompting_text = '''
    write a long postcard text without signiture to friends associate with following notes\n
    ''' + ', '.join(keywords) + '\n'
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompting_text,
        temperature=1,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=['/n/n/n/n']
    )
    return response.choices[0].text


def generate_sentence(openai,_keywords):
    keywords = _keywords
    prompting_text = '''
    Describe a photo using following notes:\n
    ''' + ', '.join(keywords) + '\n'

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompting_text,
        temperature=1,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=['/n/n/n/n']
    )
    return response.choices[0].text
