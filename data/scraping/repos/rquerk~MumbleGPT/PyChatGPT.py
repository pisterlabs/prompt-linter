import openai

ROLE = "Sei ein Mitglied vom Chaos Computer Club e.V.!"
MODEL = "gpt-3.5-turbo"

class ChatGPT:
    def __init__(self, api_key, role=ROLE):
        openai.api_key = api_key
        self.dialog = [{"role": "system", "content": role}]

    def human_request(self, request):
        self.dialog.append({"role": "user", "content":request})

        try:
            gpt_response = openai.ChatCompletion.create(model=MODEL, messages=self.dialog)
            assistant_choice = gpt_response.choices[0].message.content
        except openai.error.RateLimitError as overload:
            assistant_choice = "OpenAI Server Overloaded!\n" + str(overload)

        self.dialog.append({"role": "assistant", "content": assistant_choice})

        return assistant_choice
