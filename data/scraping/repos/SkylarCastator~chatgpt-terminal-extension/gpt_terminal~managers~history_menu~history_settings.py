import json
import os
from gpt_terminal import platform_helper
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory import ChatMessageHistory


class History:
    def __init__(self):
        self.history_path = f"{platform_helper.get_appdata_folder()}/GPT-Terminal/History/"
        if not os.path.exists(self.history_path):
            os.mkdir(self.history_path)

    def list_history_files(self):
        arr = os.listdir(self.history_path)
        return arr

    def get_history_path(self):
        return self.history_path

    def save_history(self, chat_history, file_name="history_0"):
        file_path = f"{self.history_path}/{file_name}.json"
        with open(file_path, "w") as f:
            json.dump(messages_to_dict(chat_history.messages), f, indent=4)

    def load_history(self, filename="history_0"):
        file_path = f"{self.history_path}/{filename}.json"
        with open(file_path, "r") as f:
            dict = json.load(f)
            return dict

    def delete_conversation(self, filename):
        file_path = f"{self.history_path}/{filename}.json"
        if os.path.exists(file_path):
            os.remove(file_path)

    def parse_conversation(self, chat_history):
        content_list = []
        for response in chat_history:
            content_list.append(response)
        return content_list


