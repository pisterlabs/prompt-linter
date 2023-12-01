import os

import openai
from halo import Halo
from spinners import Spinners

from brain.token_counter import count_tokens
from ears.hear import get_audio
from tools.clipboard import copy_to_clipboard_prefix
from tools.logger import logger
from tools.typewriter import typewrite

system_message = "\nSYSTEM: You are named Cartuli, " \
                 "a LLM trained by OpenAI similar to ChatGPT, " \
                 "you will give concise answers, going straight" \
                 " to the point but giving a sufficient response:\n"


def asker(text):
    audio = None
    r = None
    # Set up OpenAI API key
    openai.api_key = get_open_ai_key()
    if text is None:
        audio, r = get_audio()
    try:
        # Convert speech to text
        if text is None:
            text = r.recognize_google(audio)
            typewrite("\033[35;40mYou said: \033[0m" + f"\033[37;40m{text}\033[0m")

        return get_chat_gpt_response(text)
    except Exception as e:
        spinner = Halo(text='', spinner=Spinners['growVertical'], color='cyan')
        spinner.fail(f"Could not request results; {e}")
        spinner.stop()
        spinner.clear()


def get_open_ai_key():
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        logger("OPENAI_API_KEY environment variable is not set.")
        exit()
    return api_key


def get_chat_gpt_response(text):
    spinner = Halo(text='', spinner=Spinners['growVertical'], color='cyan')
    spinner.start()
    # Generate response from OpenAI API
    prompt = system_message + text + "?"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        # We add a question mark at the end to avoid ChatGpt trying to autocomplete our questions, and then returning wrong response, example, "Who was Elvis Presley", and it answers "'manager, Elvis Presley's manager was blablabla, which is wrong!!
        max_tokens=60,
        # top_p=0.2,
        temperature=0,
        n=1
    )
    if 'choices' in response and len(response['choices']) > 0:
        generated_text = response['choices'][0]['text'].strip()
        # Write response text to clipboard
        copy_to_clipboard_prefix(generated_text, "Response: ")
        spinner.stop()
        spinner.clear()
        logger(generated_text)
        return generated_text
    else:
        spinner.stop()
        typewrite("No response received from the API.")
        return None


def chat_with_openai(prompt):
    # response = openai.Completion.create(
    #     engine='text-davinci-003',
    #     prompt=messages + "Refactor the above code",
    #     temperature=0.7,
    #     # max_tokens=100,
    #     n=1,
    #     stop=None,
    #     top_p=1.0,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0
    # )

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        # We add a question mark at the end to avoid ChatGpt trying to autocomplete our questions, and then returning wrong response, example, "Who was Elvis Presley", and it answers "'manager, Elvis Presley's manager was blablabla, which is wrong!!
        max_tokens=count_tokens(prompt) * 2,
        # top_p=0.2,
        temperature=0,
        n=1)

    # Extract the generated message from the response
    # generated_message = response2.choices[0].text.strip().split('\n')[-1]
    generated_text = response['choices'][0]['text'].strip()

    return generated_text
