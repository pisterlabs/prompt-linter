import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file


class QuestionAnswerSystem:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def text_completion(self, text):
        resp = self.client.completions.create(
            model="text-davinci-003",
            prompt=text,
            max_tokens=256,
            temperature=1
        )

        return resp.choices[0].text

    def chat_completion(self, prompt):
        resp = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message


if __name__ == "__main__":
    obj = QuestionAnswerSystem()
    response = obj.text_completion("When the Generative AI is booming, i started learning openai's API and ChatGPT")
    print(response)