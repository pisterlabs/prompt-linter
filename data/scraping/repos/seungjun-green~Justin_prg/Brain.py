import openai
import Settings
from Resources import keys

openai.api_key = keys.ai_key

class Brain:
    def create_response(self, order, stop):
        result = ""
        count = 0
        print(stop)
        while True:
            response = openai.Completion.create(
                engine=Settings.engine,
                prompt=order,
                temperature=0.7,
                max_tokens=60,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop
            )

            count += 1
            result += response['choices'][0]['text']
            order += response['choices'][0]['text']
            if count == 3 or response['choices'][0]['text'] == '':
                break

        return result

    def create_content(self, order):
        result = ""
        count = 0
        while True:
            response = openai.Completion.create(
                engine=Settings.engine,
                prompt=order,
                temperature=0.7,
                max_tokens=60,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            count += 1
            result += response['choices'][0]['text']
            order += response['choices'][0]['text']
            if count == 3 or response['choices'][0]['text'] == '':
                break

        return result