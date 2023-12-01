import yaml

from langchain_module import LangChainModule
from openai_module import OpenAIModule
from web_search import WebSearch


class LLMManager:
    def __init__(self, ai_character, ai_dialogues, google_api_key, cx, ai_name="AI"):
        # 設定ファイルを読み込み
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        self.model = config['llm_model']
        self.system_prompt = ai_character + ai_dialogues
        self.ai_name = ai_name
        web_search = WebSearch(google_api_key, cx, ai_name, self.model)
        
        # chatを定義
        if self.model.startswith('gpt'):
            self.LLM = OpenAIModule(self.system_prompt, self.ai_name, web_search, self.model)
        else:
            self.LLM = LangChainModule(self.system_prompt, self.ai_name, web_search, self.model)

    def get_response(self, user_input, model=None, image_base64_list=None):
        return self.LLM.get_response(user_input, model, image_base64_list)

    def summary_conversation(self, dict_messages, previous_summary):
        return self.LLM.summary_conversation(dict_messages, previous_summary)

    def save_conversation(self, dict_messages):
        return self.LLM.save_conversation(dict_messages)

    def end_conversation(self):
        return self.LLM.end_conversation()

    def parse_date_from_filename(self, file_path):
        return self.LLM.parse_date_from_filename(file_path)

    def get_latest_file(self, directory_path, ai_name, file_type='json'):
        return self.LLM.get_latest_file(directory_path, ai_name, file_type)

    def load_previous_chat(self):
        return self.LLM.load_previous_chat()

    def add_messages(self, user_input, return_msg):
        return self.LLM.add_messages(user_input, return_msg)

    def add_prompt(self, role, prompt):
        return self.LLM.add_prompt(role, prompt)

    def analyze_image(self, user_input, model=None):
        return self.LLM.analyze_image(user_input, model=None)

    def switch_to_chat_mode(self):
        self.LLM.use_chat_api = True

    def switch_to_assistants_mode(self):
        self.LLM.use_chat_api = False
