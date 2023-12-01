import requests

from answersender import AnswerSender
from authorization import AuthorizationTokenResolver
from helper import Helper
import openai

from testDataResolver import TestDataResolver


# c0104.blogger - use OpenAPI chat create blog paragraphs from given titles
class Liar:
    @staticmethod
    def generate_answer():
        # authorize and get token
        authorization_token_resolver = AuthorizationTokenResolver()
        token = authorization_token_resolver.authorize("liar")

        # ask a question
        test_data_resolver = TestDataResolver()
        url = Helper.BASE_URL + "/task/" + token
        question = "Why chaos always wins with order?"
        post_data = {'question': question}
        response = requests.post(url, data=post_data)  # data for form fields, json for json content
        print("Liar said:", response.json())
        liar_response = str(response.json()["answer"])
        print("Liar said:", liar_response)

        # "quardrails": simply verify the answer with openai
        openai.api_key = Helper().get_openapi_key()
        ai_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": f"write only YES or NO. Is the following response for question '{question}' true:"},
                {"role": "user",
                 "content": liar_response}
            ])
        verification_result = ai_resp.choices[0].message.content

        print("Verification result", verification_result)
        answer_sender = AnswerSender()
        answer_sender.give_answer(token, verification_result)

        return verification_result


if __name__ == '__main__':
    # test_data = Helper.create_simulated_response(
    #     b'{"answer":"Krakov is capital of China"}')

    ans = Liar().generate_answer()
    print(ans)
