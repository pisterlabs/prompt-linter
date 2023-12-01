
import openai

import config

context = [{"role": "system", "content": "you are an english teacher who is triying to splain why a certein word doesnt go in a certein place in a sentence, respond with OK if you got it"}]
MODEL = config.MODEL
openai.api_key = config.apiKey

def init_openchat():

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=context,
        temperature=0.2,
    )

def gpt_querry(sentences_list, asked_word):

    full_text = "".join(sentences_list)
    full_querry = f"Explain me why I can not use the word {asked_word} in the place marked by _____ in the text {full_text}"

    print(full_querry)
