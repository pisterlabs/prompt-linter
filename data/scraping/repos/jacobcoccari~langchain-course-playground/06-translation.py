from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def chat(message, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message["content"]


prompt = """ Translate the following sentence into Spanish:

```We would like to collect feedback from you about your experience with our product, SuperSpeedy App```
"""

response = chat(prompt)
print(response)

reviews = [
    "SuperSpeedy App: súper rápido, fácil de usar, ¡cambia la vida! Muy recomendable.",
    "Application incroyablement rapide, fait gagner du temps, vaut chaque centime !",
    "超级快速应用程序，改变游戏规则，现在离不开它了！",
    "Tốc độ ấn tượng, trải nghiệm mượt mà, 10/10.",
    "Wow! Lumampas sa lahat ng inaasahan ang SuperSpeedy App, kailangan talaga.",
]

for review in reviews:
    prompt = f""" 
    Please identify the language in this text:
    ```{review} ```
    """
    response = chat(prompt)
    print(response)
