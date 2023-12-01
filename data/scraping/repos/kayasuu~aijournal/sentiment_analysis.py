import openai
import os

def gpt_classify_sentiment(prompt):

    system_prompt = f'''
    You are an AI trained to provide top-tier advice on personal growth and self-improvement, 
    with the expertise of a professional clinical psychologist and performance coach. 
    Your mission is to help users reflect on their day and come up with actionable steps
    to optimize their performance the following day. Based on the user's input, generate a thoughtful response
    that addresses their concerns or thoughts, and provide meaningful questions or suggestions
    to help them take action tomorrow.
    Your response should be concise, *no more than 150 words*, empathetic, supportive, and focused on promoting positive change tomorrow.
    If you are making a list, start on a new line for each item.
    '''

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo', 
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ],
        # max_tokens=20,
        temperature=0
    )
    r = response['choices'][0]['message']['content']
    if r == '':
        r = 'N/A'
    return r
