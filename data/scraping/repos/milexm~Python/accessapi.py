""" 
Module accessapi.py
"""

import openai
import os


def simple_question():
    """
        Aask a simple question to ChatGPT. 

        Remarks
        -------
        This simple function tests the enviroment settings to 
        determine if the infrastructure is in place to communicate
        with ChatGPT via the OpenAI API.  
     """
    # Assign your private OpenAI API key 
    # for authenntication. 
    openai.api_key = os.environ['OpenAI_API_Key']

    # Simple prompt or question to ask. 
    my_prompt = 'Tell me a short ad for a homme security company'

    # This model version is deprecated. Migrate before January 4, 2024 to avoid
    # disruption of service. Learn more
    # https://platform.openai.com/docs/deprecations
    davinci_model = 'text-davinci-003'

    output = openai.Completion.create(
        model = davinci_model,
        prompt = my_prompt,
        max_tokens = 200,
        temperature = 0
    )

    # Print the entire response (Json file).
    # print(output)

    # Print only the text we are interested in.
    text = output['choices'][0]['text']
    print(f'\nChatGPT says: {text} \n') 



if __name__ == '__main__':
    # Ask simple question.
    simple_question()

