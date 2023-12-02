import openai

from config import GPT_TOKEN
openai.api_key = GPT_TOKEN

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": 
            "If you find a generl question there, answer it."
            "it user asking for a help from person return ONLY 2 symbols: HP"
         },
        {"role": "user", "content": 
            "Добрый день. Можно забронировать столик на сегодня в 20:00?"},
    ]
)


print(response['choices'][0]['message']['content'])
