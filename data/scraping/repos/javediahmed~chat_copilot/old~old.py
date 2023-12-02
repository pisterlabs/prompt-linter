import openai
import os
import json
from datetime import datetime

MODELS = {
    "GPT-1": "text-gpt-1-en-12b",
    "GPT-2": "text-gpt-2-en-117b",
    "GPT-3": "text-davinci-002",
    "GPT-3.5": "text-davinci-003",
    "GPT-4": "text-davinci-004",
    "Jurassic-1 Jumbo": "text-jurassic-1-jumbo-en-175b",
    "Megatron-Turing NLG": "text-megatron-turing-nlg-345m-355b",
    "WuDao 2.0": "text-wudao-2-0-en-1.76T"
}

MAIN_MENU = {
    "1": "Chat with model",
    "2": "Submit JSON query",
    "3": "Copilot",
    "4": "Settings",
    "5": "Save/export results",
    "6": "Exit"
}

DEFAULT_SETTINGS = {
    "model": MODELS["GPT-3"],
    "query_settings": {
        "max_tokens": 60,
        "temperature": 0.5,
        "role": "user"
    },
    "menu": MAIN_MENU
}


class ChatGPT:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.settings = DEFAULT_SETTINGS.copy()
        self.history = []

    def prompt_user(self, message):
        return input(message).strip()

    def check_api_key(self):
        while not self.api_key:
            print("API Key not found.")
            self.api_key = self.prompt_user("Please enter your OpenAI API Key: ")
        openai.api_key = self.api_key

    def chat(self, query):
        self.check_api_key()
        role = self.settings["query_settings"]["role"]
        prompt = f"{role}: {query}"
        try:
            response = openai.Completion.create(
                engine=self.settings["model"],
                prompt=prompt,
                max_tokens=self.settings["query_settings"]["max_tokens"],
                temperature=self.settings["query_settings"]["temperature"]
            )
            response_text = response.choices[0].text.strip()
            self.history.append({"query": query, "response": response_text})
            print(response_text)
            return response_text
        except openai.error.APIError as e:
            print(f"OpenAI API Error: {e}")
            self.api_key = ""
            self.check_api_key()

    def query_from_json(self):
        self.check_api_key()
        json_query = self.prompt_user("Enter the JSON query: ")
        try:
            data = json.loads(json_query)
            query = data['query']
            self.chat(query)
        except (json.JSONDecodeError, KeyError):
            print("Invalid JSON query. Please try again.")

    def copilot(self):
        self.check_api_key()
        query = self.prompt_user("Enter your Copilot query: ")
        response = self.chat(query)
        self.history[-1]["copilot_response"] = response

    def update_settings(self):
        print("Current settings:")
        for key, value in self.settings["query_settings"].items():
            print(f"{key}: {value}")
        self.change_settings()

    def change_settings(self):
        print("Select setting to update:")
        for index, (key, _) in enumerate(self.settings["query_settings"].items()):
            print(f"{index + 1}. {key.capitalize()}")

        setting_choice = self.prompt_user("Enter your choice: ")
        setting_index = int(setting_choice) - 1
        settings_keys = list(self.settings["query_settings"].keys())
        if setting_index in range(len(settings_keys)):
            setting_name = settings_keys[setting_index]
            self.update_query_settings(setting_name)
        else:
            print("Invalid option. Please try again.")
            self.change_settings()

    def update_query_settings(self, setting_name):
        setting_value = self.prompt_user(f"Enter the new value for {setting_name.capitalize()}: ")
        self.settings["query_settings"][setting_name] = setting_value

    def save_results(self):
        filename = self.prompt_user("Enter the filename to save the results: ")
        with open(filename, 'w') as f:
            json.dump(self.history, f)
        print(f"Results saved to {filename}")


def process_user_input(chat_gpt, user_choice):
    for key, value in chat_gpt.settings["menu"].items():
        if user_choice == key:
            if hasattr(chat_gpt, value):
                if value == "Chat with model":
                    print("Chat session initiated")
                    chat_gpt.chat(chat_gpt.prompt_user("Enter your query: "))
                else:
                    getattr(chat_gpt, value)()
                break
            else:
                print("Invalid option. Please try again.")
                break
    else:
        print("Invalid option. Please try again.")

    return True


def main():
    chat_gpt = ChatGPT()
    print("Test GPT interface\n" + str(datetime.today().strftime('%Y-%m-%d')))
    print("\nMain Menu:")
    print("Main Menu:")
    for key, value in chat_gpt.settings["menu"].items():
        print(f"{key}. {value}")
    while True:
        user_choice = chat_gpt.prompt_user("Enter your choice: ")
        if not process_user_input(chat_gpt, user_choice):
            break


if __name__ == "__main__":
    main()
