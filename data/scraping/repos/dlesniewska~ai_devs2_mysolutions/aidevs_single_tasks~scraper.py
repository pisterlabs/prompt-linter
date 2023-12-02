import requests
from requests import Response

from helper import Helper
import openai

# c0104.blogger - use OpenAPI chat create blog paragraphs from given titles
class Scraper:

    @staticmethod
    def generate_answer(test_data):
        json_input = test_data.json()
        question = str(json_input["question"])
        article_url = str(json_input["input"])

        # retry if something went wrong
        article = requests.get(article_url)
        iteration = 0
        while article.status_code != 200 and iteration < 20:
            print("Error retrieving test data. Retrying...")
            article = requests.get(article_url)
            iteration += 1
            print(f"Test data - attempt #{iteration}")
        artile = article.text
        print("Article is: ", article.text)

        # call openai moderation api
        openai.api_key = Helper().get_openapi_key()

        ai_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": f"### fakty:### {article}"},
                {"role": "user",
                 "content": question + " Odpowiedz maksymalnie do 200 znakow "}
            ])
        json_result = ai_resp.choices[0].message.content
        return json_result

if __name__ == '__main__':

    # test_data = Response()
    # test_data.code = "OK"
    # test_data.status_code = 200
    # test_data._content = b'{"question": "Who is Alice", "article": "Alice is a software"}'
    #
    # ans = Scraper().generate_answer(test_data)
    ans = requests.get('https://zadania.aidevs.pl/text_pasta_history.txt')
    iteration = 0
    while ans.status_code != 200 and iteration < 15:
        print("Error retrieving test data. Retrying...")
        ans = requests.get('https://zadania.aidevs.pl/text_pasta_history.txt')
        iteration += 1
        print(f"Test data - attempt #{iteration}")
    print(ans)


# """blogger here
# ###
# rules:
# ###
# -i create a blog entry about pizza margherita containing 4 titles
# -list titles no deviation"""
