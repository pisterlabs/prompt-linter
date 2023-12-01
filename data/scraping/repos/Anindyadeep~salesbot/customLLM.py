import os 
import json 
import requests
from dotenv import load_dotenv
from typing import Any, Mapping, Optional, List
from langchain.llms.base import LLM 
from langchain.callbacks.manager import CallbackManager


class RapidAPILLM(LLM):
    
    load_dotenv(dotenv_path='.env/rapidapi.env')
    _url = "https://openai80.p.rapidapi.com/chat/completions"     
    _rapid_api_api_key = os.getenv("RAPID_API_KEY")
    _rapid_api_api_host = os.getenv("RAPID_API_HOST")

    @property
    def _llm_type(self) -> str:
        return "Custom LLM using RapidAPI"
    
    def _call(self, prompt : str, stop : Optional[List[str]] = None, run_manager : Optional[CallbackManager] = None) -> str:
        
        print(prompt)
        print("Length is ", len(prompt))
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": str(prompt)
                }, 
            ]
        }
        headers = {
            "content-type": "application/json",
            "X-RapidAPI-Key": self._rapid_api_api_key,
            "X-RapidAPI-Host": self._rapid_api_api_host
        }

        response =  requests.post(
            url=self._url,
            json=payload,
            headers=headers
        )

        
        return json.dumps(response.json(), indent=4) #response.json()['choices'][0]['message']['content']
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            'n' : 10000
        } 
    

if __name__ == '__main__':
    llm = RapidAPILLM()
    print(llm("Who are the competitors of Corridor Platforms?"))