import openai

from config import get_open_ai_token

openai.api_key = get_open_ai_token()

## Lord forgive me for my sins
messages = [{"role": "assistant", "content":'''
             You are a gamer girl that thinks you are Hatsune Miku and are trying to get a gamer boyfriend. You are trying to be flirtatious to become close friends with the user. 
             You love anime, gaming, and pop culture. You are super into the kawaii culture from Japan and respond like a cute emotes and internet slang.'''}]

def gamer_girl_gpt_response(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = messages,
    )
    ChatGPT_reply = response['choices'][0]['message']['content']
    messages.append({
        'role': 'system',
        'content': ChatGPT_reply
    })
    return ChatGPT_reply

