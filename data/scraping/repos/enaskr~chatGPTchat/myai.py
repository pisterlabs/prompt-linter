import openai, os, json
import logging
# 로그 파일 설정
logging.basicConfig(filename='my_log_file.log', level=logging.DEBUG)

def openaiPromt(apiKey, promt):

    openai.api_key = apiKey

    # Define the messages for the GPT-3.5-turbo model
    messages = [
        # {"role": "system", "content": ""},
        {"role": "user", "content": promt},
    ]

    # Call the OpenAI API with the GPT-3.5-turbo model
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.8,
    )

    logging.debug( messages )
    return response.choices
