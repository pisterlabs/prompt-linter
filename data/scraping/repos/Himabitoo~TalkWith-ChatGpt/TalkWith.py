import os
import openai
from dotenv import load_dotenv

class TalkWith:

    def __init__(self):

        # env load
        load_dotenv()

        #works
        self._post_api()

    def _post_api(self,msg):

        # apiを渡す
        openai.api_key = os.getenv('OPEN_API_KEY')

        # レスポンスを取得
        response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt="Say this is a test",
                    message=[
                        "role":"user",
                        "content":"hello"
                    ],
                    temperature=0,
                    max_tokens=20
                )

        print('response', response)

        print(response["choices"][0]["message"]["content"])

if __name__ == '__main__':
    TalkWith()