from typing import Union, Optional, List, Tuple, Dict, Any
import os
import json
from ChatHistory import ChatHistory, MalformedSaveDictError
import time 
import datetime
import openai
import tiktoken 
import my_secrets as secrets
class SaveChatHistory(ChatHistory):
    def __init__(self,  max_model_tokens= 8000, max_completion_tokens=1000, token_padding = 500 , chat_log_folder = "chat_saves", ):
        super().__init__(max_model_tokens, max_completion_tokens, token_padding)
        self.chat_log_folder = chat_log_folder
        self.loaded_from_save = False
        self.loaded_from = None
        
       
    def make_save_dict(self):
        """Make a dictionary that can be saved to a json file"""
        save_dict = {
            "sys_prompt": self._sys_prompt,
            "token_info": {
                "max_model_tokens": self.max_model_tokens,
                "max_completion_tokens": self.max_completion_tokens,
                "token_padding": self.token_padding,
                },
            "full_chat_log": self.full_chat_log,
            "trimmed_chat_log": self.trimmed_chat_log,
            "trimmed_messages": self.trimmed_messages,

        }
        return save_dict
    def check_save_dict(self, save_dict: dict) -> None:
        "Checks if save_dict is valid, raises MalformedSaveDictError if not"
        if not isinstance(save_dict, dict):
            raise TypeError("save_dict must be a dict")
        if not "sys_prompt" in save_dict:
            raise MalformedSaveDictError("sys_prompt")
        if not "token_info" in save_dict:
            raise MalformedSaveDictError("token_info")
        else: 
            for key in ["max_model_tokens", "max_completion_tokens", "token_padding"]:
                if not key in save_dict["token_info"]:
                    raise MalformedSaveDictError(key)
        if not "full_chat_log" in save_dict:
            raise MalformedSaveDictError("full_chat_log")
        if not "trimmed_chat_log" in save_dict:
            raise MalformedSaveDictError("trimmed_chat_log")
        if not "trimmed_messages" in save_dict:
            raise MalformedSaveDictError("trimmed_messages")
    def load_save_dict(self, save_dict: dict) -> None:
        """Load a save_dict into the chat history"""
        self.check_save_dict(save_dict)
        self.sys_prompt = save_dict["sys_prompt"]
        self.max_model_tokens = save_dict["token_info"]["max_model_tokens"]
        self.max_completion_tokens = save_dict["token_info"]["max_completion_tokens"]
        self.token_padding = save_dict["token_info"]["token_padding"]
        self.full_chat_log = save_dict["full_chat_log"]
        self.trimmed_chat_log = save_dict["trimmed_chat_log"]
        self.trimmed_messages = save_dict["trimmed_messages"]
        self.work_out_tokens()
        self.trim_chat_log()
        self.loaded_from_save = True
    def add_file_path(self, file_name: str) -> str:
        """ Adds the chat_log_folder to the file_name and .json if not already there"""
        if not file_name.endswith(".json"):
            file_name += ".json"
        return os.path.join(self.chat_log_folder, file_name)
    def get_file_name(self, file_path: str) -> str:
        """ Removes the path and .json from the file_path"""
        name =  os.path.basename(file_path)
        if name.endswith(".json"):
            name = name[:-5]
        return name
    def save_chat_log(self, file_name: str, overwrite: bool = False) -> None:
        """
        Saves the chat log to a json file in the chat_log_folder
        """
        self.check_sys_prompt()
        if not os.path.exists(self.chat_log_folder):
            os.mkdir(self.chat_log_folder)
        if not overwrite and os.path.exists(self.add_file_path(file_name)):
            return False
        file_path = self.add_file_path(file_name)
        save_dict = self.make_save_dict()
        with open(file_path, "w") as file:
            json.dump(save_dict, file)
        
    def load_chat_log(self, file_name: str):
        """ Loads a chat log from a json file in the chat_log_folder"""
        file_path = self.add_file_path(file_name)
        with open(file_path, "r") as file:
            save_dict = json.load(file)
        self.load_save_dict(save_dict)
        self.loaded_from = file_name
    def get_chat_log_names(self, string: bool = False) -> list[str]:
        """Get the names of all the chat logs in the chat_log_folder"""
        names = []
        for file in os.listdir(self.chat_log_folder):
            if file.endswith(".json"):
                names.append(self.get_file_name(file))
        if string:
            names = "\n".join(names)
        return names
    def __repr__(self):
        """Return a string that can be used to recreate the chat log"""
        return "SaveChatHistory( max_model_tokens = {max_model_tokens}, max_completion_tokens = {max_completion}, token_padding = {padding}, chat_log_folder = {chat_log_folder}) \n" + \
            "self.sys_prompt = {system_prompt}".format(system_prompt = self.sys_prompt, max_model_tokens = self.max_model_tokens, max_completion = self.max_completion_tokens, padding = self.token_padding, chat_log_folder = self.chat_log_folder) + \
            "\nself.load_chat_log({file_name})".format(file_name = self.loaded_from) + \
            super().__repr__()


        
            
        
        
    


class GPTChat(SaveChatHistory):
    def __init__(self, api_key: str , max_model_tokens= 8000, max_completion_tokens=1000, token_padding = 500, chat_log_folder = "chat_saves", model_name = "gpt-4", temperature = 0.9, top_p = 1.0, frequency_penalty = 0.0, presence_penalty = 0.0,):
        super().__init__(max_model_tokens, max_completion_tokens, token_padding, chat_log_folder)
        self.api_key = api_key
        self.model_name = model_name
        self.model_params = {
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": self.max_completion_tokens,

        }
    def _send_chat_history_to_openai(self):
        self.trys = 0
        while self.trys < 4: 
         
         openai.api_key = self.api_key
         try:
            response = openai.ChatCompletion.create(
                model = self.model_name,
                messages = self.finished_chat_log,
                **self.model_params
                )
            response = response.choices[0].message.content
            return response
         except openai.OpenAIError as error:
             print("OpenAI Error: ", error)
             self.trys += 1
             print("Try: ", str(self.trys), " of 3")
             print("Trying again...")
             time.sleep(4)

        print("Failed to get response from OpenAI")
    @ChatHistory.user_message.setter
    def user_message(self, message: str) -> None:
        """Adds a user message to chat log """
        self.check_sys_prompt()
        self.add_message(message, "user")
        self.assistant_message = self._send_chat_history_to_openai()
        self.trim_chat_log()
    def send_message(self, message: str) -> str:
        """Sends a message to the chat log and returns the response"""
        self.user_message = message
        return self.assistant_message
        

        
        
chat = GPTChat(secrets.openai_key)
chat.sys_prompt = "You are an ai assistant being tested "
chat.user_message = "Hello"
print(chat.assistant_message)
    