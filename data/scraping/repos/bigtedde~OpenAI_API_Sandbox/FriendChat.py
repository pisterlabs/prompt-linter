# Ted Lawson 2/20/23

import openai

class FriendChat:
    def __init__(self, human_name="You", ai_name="Friend"):
        self.convo: str = ""
        self.human_name: str = human_name
        self.ai_name: str = ai_name
        # integrate 'grammar correction' api for user input

    def start_chat(self):
        print("\nYou are now in a chat with your AI friend! Talk about anything.\n")
        while True:
            self.add_human_response()
            self.add_ai_response()
            print(f"\nBEGIN CONVO"
                  f"{self.convo}\n"
                  f"END CONVO\n")

    def get_human_response(self) -> str:
        response = input(f"{self.human_name}: ")
        if response != "quit":
            return response
        quit()

    def get_ai_response(self):
        # convo can not be longer then 4097 token
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"{self.convo}\n",
            temperature=.5,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0,
            stop=["You:"]
        )
        return response

    def add_ai_response(self):
        response = self.get_ai_response()
        text = response["choices"][0]["text"]
        print(f"{self.ai_name}: {text}")
        self.convo += text

    def add_human_response(self):
        response = self.get_human_response()
        text = f"\n{self.human_name}: {response} " \
               f"\n{self.ai_name}: "
        self.convo += text
