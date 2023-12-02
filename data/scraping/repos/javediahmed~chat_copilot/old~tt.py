import openai
import os
import json
from datetime import datetime

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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

DEFAULT_SETTINGS = {
    "Model": "GPT-3",
    "Query Settings": {
        "Max Tokens": 60,
        "Temperature": 0.5,
        "Role": "user"
    },
    "Menu": {
        "1": "Chat",
        "2": "Copilot",
        "3": "Update settings",
        "4": "Save results",
        "5": "Exit"
    }
}

class ChatGPT:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.settings = DEFAULT_SETTINGS.copy()
        self.history = []

    def prompt_user(self, message):
        return input(message).strip()

    def check_api_key(self):
        while not self.api_key:
            print("API Key not found.")
            self.api_key = self.prompt_user("Please enter your OpenAI API Key: ")
        openai.api_key = self.api_key

    def chat(self):
        self.check_api_key()
        model_name = self.settings['Model']
        model_value = MODELS[model_name]
        print(f'\nChatting with {model_name} ({model_value})')
        while True:
            query = self.prompt_user("Enter your query ('f' to submit by file or 'x' to exit): ")
            if query.lower() == 'f':
                json_query = self.prompt_user("Enter the path to JSON file: ")
                try:
                    with open(json_query, 'r') as file:
                        data = json.load(file)
                        query = data['query']
                except (json.JSONDecodeError, KeyError, FileNotFoundError):
                    print("Invalid JSON query. Please try again.")
                    continue
            elif query.lower() == 'x':
                break
            role = self.settings["Query Settings"]["Role"]
            prompt = f"{role}: {query}"
            query_settings = self.settings["Query Settings"]
            try:
                response = openai.Completion.create(
                    engine=model_value,
                    prompt=prompt,
                    max_tokens=query_settings["Max Tokens"],
                    temperature=query_settings["Temperature"]
                )
                response_text = response.choices[0].text.strip()
                self.history.append({"query": query, "response": response_text})
                print(response_text)
            except openai.error.APIError as e:
                print(f"OpenAI API Error: {e}")
                self.api_key = ""
                self.check_api_key()

    def copilot(self):
        self.check_api_key()
        self.chat()
        self.history[-1]["copilot_response"] = self.history[-1]["response"]

    def update_settings(self):
        print("Which setting would you like to change?")
        self.display_settings(self.settings)
        self.change_settings(self.settings)

    def display_settings(self, settings, level=1, prefix=""):
        count = 1
        for key, value in settings.items():
            if key != "Menu":
                if isinstance(value, dict):
                    print(f"{count}. {prefix}{key}:")
                    count += 1
                    self.display_settings(value, level=level+1, prefix=" " * 4)
                else:
                    print(f"{count}. {prefix}{key}: {value}")
                    count += 1

    def change_settings(self, settings):
        setting_index = self.prompt_user("Enter the setting index to update (0 to go back): ")
        if setting_index == '0':
            return

        keys = list(settings.keys())
        if setting_index in map(str, range(1, len(keys) + 1)):
            selected_key = keys[int(setting_index) - 1]
            if isinstance(settings[selected_key], dict):
                self.update_subsettings(settings[selected_key])
            else:
                new_value = self.prompt_user(f"Enter the new value for {selected_key}: ")
                settings[selected_key] = new_value
        else:
            print("Invalid setting index. Please try again.")

    def update_subsettings(self, settings):
        print("Which sub-setting would you like to change?")
        self.display_settings(settings)
        self.change_settings(settings)

    def save_results(self):
        filename = self.prompt_user("Enter the filename to save the results: ")
        with open(filename, 'w') as f:
            json.dump(self.history, f)
        print(f"Results saved to {filename}")

    def run(self):
        print("Test GPT interface\n" + str(datetime.today().strftime('%Y-%m-%d')))
        menu = self.settings["Menu"]
        while True:
            print("\nMain Menu:")
            for key, value in menu.items():
                print(f"{key}. {value}")
            user_choice = self.prompt_user("Enter your choice: ")
            if user_choice in menu:
                if user_choice == "1":
                    self.chat()
                elif user_choice == "2":
                    self.copilot()
                elif user_choice == "3":
                    self.update_settings()
                elif user_choice == "4":
                    self.save_results()
                elif user_choice == "5":
                    break
            else:
                print("Invalid option. Please try again.")

if __name__ == "__main__":
    chat_gpt = ChatGPT()
    chat_gpt.run()
