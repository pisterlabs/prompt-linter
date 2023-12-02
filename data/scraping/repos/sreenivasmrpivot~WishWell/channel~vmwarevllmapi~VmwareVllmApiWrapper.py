import json
# import openai
from openai import OpenAI
from helper import get_model_path

from models import Wish

class VmwareVllmApiWrapper:
    
    cred_file = "./creds.json"
    endpoint = "https://vllm.libra.decc.vmware.com/api/v1"

    def __init__(self, wish: Wish):
        self.wish = wish
        self._load_creds()
        self._initialize_openai_client()
        
    def _load_creds(self):
        with open(VmwareVllmApiWrapper.cred_file, "r") as creds:
            api_key = json.loads(creds.read())["api_key"]
            self.api_key = api_key
    
    def _initialize_openai_client(self):
        self.client = OpenAI(api_key = self.api_key, base_url = VmwareVllmApiWrapper.endpoint) 
    
    def run(self):
        # Option 1: You may use chat completions to access the LLMs as below 
        # completions = self.client.chat.completions.create(
        #     model=get_model_path(self.wish),
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": self.wish.whisper,
        #         },
        #     ],
        # )        

        # Option 2: You may directly use the completions to access the LLMs as below
        completions = self.client.completions.create(
            model=get_model_path(self.wish),
            prompt=self.wish.whisper
        )        
        grant = completions.choices[0].text
        return grant
