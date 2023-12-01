from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.StreamHandler import StreamHandler
from pathlib import Path
import json

class Worker:
    def __init__(self, model):
        self.model = model
        self.config = self.load_config()
        
    def load_config(self):
        config_path = Path(__file__).resolve().parent.parent / 'configs' / 'config.json'
        with config_path.open('r', encoding='utf-8') as f:
            config = json.load(f)
        return config



