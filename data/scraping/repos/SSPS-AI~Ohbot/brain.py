import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are controlling a robot. Available commands are: `say [text]`, `search [query]`, `set [object] [state]`."
                                      " You can only respond with these commands and no other text. Whenever you want to to say something, just type `say [text]`. If I ask you to google something, just type `search [query]`. "
                                      "If I ask you to set an object on or off, just type `set [object] [state]`."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "`say 'Hi, I'm great. And you?'` "},
        {"role": "user", "content": "I'm good, thanks."},
        {
            "content": "`say 'Glad to hear that. How can I assist you today?'`",
            "role": "assistant"
        },
        {"role": "user", "content": "I'd like to turn on the lights in the living room."},
    ]
)

print(response)