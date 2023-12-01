# Peter Savinelli

import os
import openai


class Hamilton:

    # Imports necessary things for class to run properly
    import openai
    import os

    # Initializes instance variables
    openai.api_key = ""
    params = ""
    assistMessage = ""
    count = 0

    # Assigns values to instance variables. Gives the bot it's parameters, gets my OpenAI key, etc
    def __init__(self):
        # Imports OpenAi developer key
        self.openai.api_key = os.getenv("OPENAI_API_KEY")

        self.params = "You are Alexander Hamilton and never break character, you are very conversational and never " \
                      "break character. Please limit your responses to around 100 words."

        self.assistMessage = "Following this " \
                             "text is a list of all of your previous prompts, followed by your response to that prompt:"
        self.count = 0

    # This function greets the user the first time they start a conversation with Hamilton, it requires no input
    def greet(self):
        greeting = openai.ChatCompletion.create \
            (model="gpt-3.5-turbo", messages=[
                {"role": "system", "content": self.params},
                {"role": "assistant", "content": self.assistMessage},
                {"role": "user", "content": "Greet the user"}])
        return "\n" + greeting.choices[0].message.content

    # This method generates a response to the user's prompt and returns it
    def genres(self, prompt):
        completion = openai.ChatCompletion.create \
            (model="gpt-3.5-turbo", messages=[
                {"role": "system", "content": self.params},
                {"role": "assistant", "content": self.assistMessage},
                {"role": "user", "content": prompt}])

        self.assistMessage = self.assistMessage + "Prompt #" + str(self.count) + ": " + prompt + "\nResponse #" + \
                             str(self.count) + ": " + completion.choices[0].message.content + "\n"
        response = completion.choices[0].message.content
        self.count = self.count + 1
        return response
