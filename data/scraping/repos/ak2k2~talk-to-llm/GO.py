# Property of JargonBots
# Written by Armaan Kapoor on 12-26-2022

from brain.wrapper import OpenaiRequest
from ears.listen import stt_google
from mouth.talk import speak
import time


class Conversation:
    def __init__(self, starting_environment, bot_name, voice="default"):
        self.bot_name = bot_name
        self.environment = str(starting_environment)

    def get_environment(self):
        return str(self.environment)

    def user_append(self):
        from Levenshtein import distance as levenshtein_distance

        print("\n")
        # user_text = str(input("You: "))
        vox = None
        while not vox:
            vox = stt_google()
        print(vox)

        if levenshtein_distance(vox, "show me the weather") < 5:
            import requests

            C = "Millburn"
            url = "https://wttr.in/{}".format(C)
            try:
                data = requests.get(url)
                T = data.text
            except:
                T = "Error Occurred"

            req = OpenaiRequest(T[0:1000] + "\nSummarize the weather in one sentence:")
            T = req.return_tokens(0.3, 200, 0.0, None)
            print(T)
            vox = "Say the following: " + T

        if levenshtein_distance(vox, "what time is it") < 5:
            T = time.strftime("%I:%M %p")
            vox = "Say the following: " + T

        print("You: " + vox)
        self.environment += "\nMe: " + vox
        return vox

    def bot_append(self):
        injection = self.environment + "\n" + self.bot_name
        req = OpenaiRequest(injection)
        self.toks = req.return_tokens(1.2, 2000, 0.0, ["Me:", self.bot_name + ":"])
        print("\n")
        self.environment = injection + self.toks

    def display_response(self):
        print(self.bot_name + self.toks)
        speak(self.toks, voice)

    def store_environment(self, file_name):
        rate = input("\nRate the conversation on a scale of 1-10: ")
        with open(file_name, "w") as f:
            f.write(self.environment)
            f.write("\n\nRating: " + rate)

    def tick(self):
        user_text = self.user_append()

        while user_text != "exit":
            self.bot_append()
            self.display_response()
            user_text = self.user_append()

        print("\nExiting Chat With {}...".format(self.bot_name[:-2:]))


Name = "DONALD TRUMP"
voice = "TRUMP"

cv = Conversation(
    voice=voice,
    starting_environment=f"The following is a conversation with {Name}.\n",
    bot_name=f"{Name}: ",
)

speak(f"Hi. I'm {Name}", voice)
time.sleep(0.3)
cv.tick()
