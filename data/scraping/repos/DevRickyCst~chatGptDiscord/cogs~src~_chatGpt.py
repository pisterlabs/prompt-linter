import openai


class Gpt:
    def __init__(self, api_key) -> None:
        openai.api_key = api_key

    def call_chat(self, content):
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": content}]
        )
        return chat_completion.choices[0].message.content

    def call_image(self, content):
        response = openai.Image.create(prompt=content, n=1, size="1024x1024")
        return response["data"][0]["url"]

    def call_traduction(self, lang, content):
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "Bonjour Gpt peux tu me tranduire en {lang} la phrase suivante : {content}".format(
                        lang=lang, content=content
                    ),
                }
            ],
        )
        return chat_completion.choices[0].message.content
