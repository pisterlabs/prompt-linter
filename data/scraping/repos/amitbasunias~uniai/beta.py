import openai

OPENAI_API_KEY = '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'

openai.api_key = OPENAI_API_KEY


def headline (headprompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="{}" .format(headprompt),
        temperature=0.9,
        max_tokens=3400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    if 'choices' in response:

        answer= response['choices'][0]['text']
    return answer

