from langchain.llms.base import LLM, BaseLanguageModel
from typing import Optional, List, Mapping, Any
import requests
import json
from collections import deque
class VicunaLLM(LLM):
    pload = {
        "model": "vicuna-13b",
        "prompt": "",
        "temperature": 0.0,
        "max_new_tokens": min(600, 1536),
        "stop":"\n### Human:"
    }
    model:str = "vicuna-13b"
    server_url:str = "http://localhost:21002"
    headers = {"User-Agent": "fastchat Client"}
    @property
    def _llm_type(self) -> str:
        
        return "vicuna"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        self.pload["prompt"] = "### Human:"+prompt+"\n\n### Assistant:"
        if stop is None or len(stop) == 0:
            stop = ["\n### Human:"]
        self.pload["stop"] = stop[0]
        self.pload["model"] = self.model
        return self.ask_chatbot_without_stream(self.pload)

      
    def ask_chatbot_without_stream(self, pload):
        response = requests.post(
            self.server_url + "/worker_generate_stream",
            json=pload, stream=False, timeout=10, headers=self.headers
        )
        output = ""
        last_but_one_chunk = None
    
        for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
            if len(chunk) > 0:
                last_but_one_chunk = chunk
    
        # Process the last but one chunk
        if last_but_one_chunk:
            data = json.loads(last_but_one_chunk.decode("utf-8"))
            #skip_echo_len = len(pload['prompt'].replace("</s>", " ")) + 1
            output = data["text"].strip()
        return output
