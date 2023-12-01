import openai
import os

class CreateQuiz:
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    def __int__(self):
        pass

    @staticmethod
    def create_quiz(text):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant. Create a 10 question multiple choice quiz based off the text. Have the answers at the end of the quiz."},
                {"role": "user", "content": f"{text}"}
            ]
        )
        quiz = response['choices'][0]['message']['content'].strip()
        with open("openai_quiz.txt", "w") as file:
            file.write(quiz)

        print("Quiz created.")
