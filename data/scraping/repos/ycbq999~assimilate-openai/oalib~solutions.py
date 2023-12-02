"""Library with OpenAI API solution as functions

References:
For building code: https://platform.openai.com/docs/guides/code/intorduction

"""

import os
import openai




def submit_question(text):
    """This submits a question to the OpenAI API"""

    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = text

    result = openai.Completion.create(
        prompt=prompt,
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model="text-davinci-002",
    )["choices"][0]["text"].strip(" \n")
    return result

#build a function that converts a text into code in any language
def create_code(text, language="python3"):

    """This submits a comment to the OpenAI API to create code in any languag
        
        Example:
            language = 'Python3' 
            text = f"Calculate the mean distance between an array of points"
            create_code(text, language)
        
    """
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{text} using {language}."},
    ]
    )

    result = completion.choices[0].message["content"]
    print(result)

    return result



