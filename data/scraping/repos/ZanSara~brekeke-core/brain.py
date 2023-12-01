import os
import openai
from datetime import datetime

openai.api_key = os.getenv("OPENAI_API_KEY")

END_TOKEN = "~~END OF CONVERSATION~~"

class Brain:

    def __init__(self):
        self.reset_history()

    def reset_history(self):
        self.history = f"""
Your name is Brekeke. You're having a casual conversation with a person.
The time is {datetime.now().strftime("%H:%M ")}. 
The date is: {datetime.now().strftime("%A, %d %B %Y")}.
You're in Lisbon.

When you're sure the person won't reply anymore, you should say '{END_TOKEN}'.
You can only generate your own answers, not the person's.
"""

    def reply(self, question):
        self.history = f"{self.history}\nPerson: {question}\nYou: "

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=self.history,
            temperature=0.5,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        reply = response['choices'][0]['text']
        print(self.history + reply)

        if END_TOKEN in reply:
            self.reset_history()
        else:
            self.history += reply
        return reply.replace(END_TOKEN, ""), END_TOKEN in reply


if __name__ == "__main__":
    print(Brain().reply("Hello, I'm Sara!"))
