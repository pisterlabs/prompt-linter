import json

import LlamaIndex
CHAT_HISTORY_PATH = "chat_history.json"

class ChatObi:
    def __init__(self):
        self.chat_history = []

    def query_index(self, input_query):
        prompt = "provide brief answers to questions based on specific private company context, if the question cannot be answered, say \" Hrrrmmmm. Know this i do not, check the DXL wikis you must: https://dev.azure.com/vfuk-digital/Digital/_wiki/wikis/Digital%20X.wiki\" \n"\
            .join([f"{message['role']}: {message['content']}" for message in self.chat_history[-2:]])
        prompt += f"\n User: In vodafone {input_query}"

        response = LlamaIndex.loadIndex().query(prompt, mode="default", response_mode="tree_summarize")
        message = {"role": "assistant", "content": response.response}
        self.chat_history.append({"role": "user", "content": input_query})
        self.chat_history.append(message)
        return response.response

    def load_chat_history(self, filename):
        try:
            with open(filename, 'r') as f:
                self.chat_history = json.load(f)
        except FileNotFoundError:
            pass

    def save_chat_history(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.chat_history, f)


    def clear_chat_history(self):
        self.chat_history.clear()
        self.save_chat_history(CHAT_HISTORY_PATH)
