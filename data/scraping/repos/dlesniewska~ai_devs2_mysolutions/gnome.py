import openai
from langchain.llms.openai import OpenAI

from helper import Helper

# C04003 gnome - rozpoznaj kolor czapki gnomów na obrazku lub zwróć ERROR
# używamy modelu gpt4 vision
class Gnome:
    @staticmethod
    def generate_answer(test_data):
        img_url = str(test_data.json()["url"])
        openai.api_key = Helper().get_openapi_key()

        ai_resp = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "jaki jest kolor czapki gnoma/skrzata. odpowiedź: 'znaleziony_kolor' lub  jeżeli nie jest to gnom to odpowiedz 'ERROR' "},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        ai_answer = (ai_resp.choices[0].message.content)
        print(ai_answer)
        return ai_answer


if __name__ == '__main__':
    test_data = Helper.create_simulated_response(
        b'{"img_url":"https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}')

    ans = Gnome().generate_answer(test_data)
    print(ans)




