import requests
import os, json
import time
import openai
from dotenv import load_dotenv
from util import *

load_dotenv()
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AZURE_URL = os.getenv('AZURE_URL')
AZURE_API_KEY = os.getenv('AZURE_API_KEY')

openai.api_key  = OPENAI_API_KEY

class Sherlock:
    def __init__(self, config_path, **kwargs):
        if config_path is not None:
            self._init_from_json(config_path)
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        history = {}
        q_history = [build_content(self.get_first_prompt())]
        k_history = []
        history['q_history'] = q_history
        history['k_history'] = k_history

        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)

    def _init_from_json(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, encoding='UTF8') as f:
                data = json.load(f)
            # init config
            self._init_from_dict(data)
        else:
            raise FileNotFoundError("no config")

    def _init_from_dict(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)
    
    def get_system_prompt(self):
        history = json.load(open(self.history_path))
        k_histoy = history['k_history']
        searched_keywords = ';'.join(k_histoy)
        if 'summary' in history:
            summ_text = history['summary']
        else:
            summ_text = ""
        sys_prompt, _ = get_prompt("herstory_v1.1")
        sys_prompt = sys_prompt.replace('[__KEYWORDS__]', searched_keywords)
        sys_prompt = sys_prompt.replace('[__SUMMARY__]', summ_text)
        
        return sys_prompt

    def get_first_prompt(self):
        #f = open(self.first_user_prompt_path, 'r', encoding='UTF8')
        #first_prompt = f.read()
        _, first_prompt = get_prompt("herstory_v1.1")
        return first_prompt

    def truncate(self, q_histroy, n=-20):
        result = []
        result += q_histroy[n:]
        return result
    
    def get_gpt_response_azure(self):
        history = json.load(open(self.history_path))
        q_history = self.truncate(history['q_history'], -12)
        messages = [{'role': 'system', 'content': self.get_system_prompt()}]
        messages += q_history
        headers = {'Content-Type': 'application/json','api-key': AZURE_API_KEY}
        
        payload = {
        'messages': messages,
        'temperature' : self.temperature,
        'top_p': self.top_p,
        'frequency_penalty' : self.frequency_penalty,
        'presence_penalty' : self.presence_penalty
        }
        response = requests.post(AZURE_URL, headers=headers, data=json.dumps(payload))
        response = response.json()
        
        try:
            result = response['choices'][0]['message']['content'].strip()
        except:
            result = None
        return result
    
    def get_gpt_response(self):
        for _ in range(3):
            history = json.load(open(self.history_path))
            q_history = self.truncate(history['q_history'])
            messages = [{'role': 'system', 'content': self.get_system_prompt()}]
            messages += q_history
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-4', 
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty
                )
                result = response['choices'][0]['message']['content'].strip()
                return result
            except:
                print(f"INVALID GPT RESULT, TRYING AGIN {_}")
                time.sleep(5*_)
        return None
        
    def save_history(self, contents, role):
        history = json.load(open(self.history_path))
        q_history = history['q_history']
        contents = build_content(contents, role)
        q_history.append(contents)
        history['q_history'] = q_history
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)

    def save_keyword(self, keyword):
        history = json.load(open(self.history_path))
        k_history = history['k_history']
        k_history.append(keyword)
        history['k_history'] = k_history
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)
    
    def get_history(self):
        return json.load(open(self.history_path))
    
    
    def pop_history(self):
        history = self.get_history()
        history['q_history'] = history['q_history'][:-1]
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)

    def save_summary(self, summ_text):
        history = self.get_history()
        history['summary'] = summ_text
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4, ensure_ascii=False)