import openai
from key import api_key
from gpytranslate import SyncTranslator

t = SyncTranslator()

openai.api_key = api_key


def dialog_with_yulia(message, name):
    user_message_lang = t.detect(message)
    user_message_trans = t.translate(message, targetlang="en")
    user_message_en = user_message_trans

    with open('memory.txt', 'a') as f:
        f.write(f"{name}: {user_message_en} \n")

    with open('memory.txt', 'r') as file:
        memory = file.read()

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=memory,
        temperature=0.25,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0,
    )
    ai_text = response["choices"][0]["text"]

    with open('memory.txt', 'a') as f:
        f.write(f"{ai_text} \n")

    ai_text_trans = t.translate(ai_text, targetlang=user_message_lang)

    return ai_text_trans["text"]
