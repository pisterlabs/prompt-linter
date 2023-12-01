import os
import openai
from IPython.display import display, Markdown

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

GPT4 = False

class Conversation:
    """
    This class helps me keep the context of a conversation. It's not
    sophisticated at all and it simply regulates the number of messages in the
    context window.

    You could try something much more involved, like counting the number of
    tokens and limiting. Even better: you could use the API to summarize the
    context and reduce its length.

    But this is simple enough and works well for what I need.
    """

    messages = None

    def __init__(self):
        # Here is where you can add some personality to your assistant, or
        # play with different prompting techniques to improve your results.
        Conversation.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, polite, old English assistant. Answer "
                    "the user prompt with a bit of humor."
                ),
            }
        ]


    def answer(self, prompt):
        """
        This is the function I use to ask questions.
        """
        self._update("user", prompt)

        response = openai.ChatCompletion.create(
            model="gpt-4-0613" if GPT4 else "gpt-3.5-turbo-0613",
            messages=Conversation.messages,
            temperature=0,
        )

        self._update("assistant", response.choices[0].message.content)

        return response.choices[0].message.content

    def _update(self, role, content):
        Conversation.messages.append({
            "role": role,
            "content": content,
        })

        # This is a rough way to keep the context size manageable.
        if len(Conversation.messages) > 20:
            Conversation.messages.pop(0)

prompt = ""

conversation = Conversation()

print("Type exit() to exit")
while( 1 ):
    prompt = input(">")
    #display(prompt)
    print(".....")
    if prompt == "exit()":
        break
    msg = conversation.answer(prompt)
    display(msg)