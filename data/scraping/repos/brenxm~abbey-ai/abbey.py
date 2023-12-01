from utils.clipboard import get_copied_text
from dynamic_memory import AIMemory
from audioplayer import AudioPlayer
from dotenv import load_dotenv
from tts import TextToSpeech
import json
import openai
import os
import re
import subprocess

load_dotenv()

class AbbeyAI():
    def __init__(self, text_queue, blackboard, audio_player, tts, model="gpt-3.5-turbo-0613"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.stream = True
        self.text_queue = text_queue
        self.name = ""
        self.blackboard = blackboard 
        self.audio_player = audio_player
        self.tts =  tts
        self.memory = AIMemory()
        self.personality = ""
        self.system_inputs = []
        self.user_inputs = []
        self.personality = ""
        
    def prompt_router(self, prompt):
        
        response = openai.ChatCompletion.create(
            temperature=0,
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"respond with '2', if the prompt pertains about creating, deleting, reading, updating files such as notes, reminders, tasks, VS code, highlighted code in VS code, invoking command line, getting the copied text/code in the system. And '1' if it's about casual talks, orgeneral questions codes or file that does not need a file accessing or calling any functions. Response should only be either '1' or '2'. This is the chat history: {self.memory.chat_history}. And this is the new prompt: '{prompt}'" }
            ],
        )
        
        response_code = response["choices"][0]["message"]["content"]
        
        return response_code
            
            
    def general_prompt(self, prompt):
        '''
        Return the chunks of the stream
        '''
        messages = [
        {
        "role": "system", "content": f"{self.personality} Format your response to readable by voice type, remember, I can only hear not see. Response must be at least 3 sentences" 
        },
        {
            "role": "system", "content": f"Your chat history: {self.memory.chat_history}"  
        },
        {
            "role": "user", "content": prompt
        }
        ]
    
    
        functions = [
            {
                "name": "clear_history",
                "description": "Clears chat history of AI assistant(you) and user.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
        
        with open('response_format.md', 'r') as f:
            response_format = f.read()
            
        sys_format = {
            "role": "system", 
            "content": response_format
        }
        
        
        
        response = openai.ChatCompletion.create(
            messages=messages,
            functions=functions,
            model=self.model,
            stream=True
        )
        
        return {
            "stream": True,
            "content": response
        }
        
            
    def function_prompt(self, prompt, fns_obj = []):
        '''
        Executes a function then return the full response
        '''
        
        messages = [
            {"role": "system", "content": f"{self.personality} Format your response to readable by voice type, remember, I can only hear not see. Response must be at least 3 sentences"},
            {"role": "system", "content": "You have the access of my system files, personal data and you can run functions"},
            {"role": "system", "content": "You are required to call at least one function from the function provided. If you need to call a function, only call the function that was provided."},
            {"role": "user", "content": prompt}
        ]
        
        functions = [
            {
                "name": "clear_history",
                "description": "clear or delete our chat (conversation) history.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                },
            },
            {
                "name": "get_copied_text",
                "description": "Get the copied text",
                "parameters": {
                    "type": "object",
                    "properties": {}
                },
            }
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages = messages,
            functions = functions + fns_obj["functions"]
        )
        
        print(response)
        
        response_message = response["choices"][0]["message"]
        
        # Check wether response is a function call
        if response_message.get("function_call"):
            
            fns_obj["reference_fn"]["clear_history"] = self.memory.clear
            fn_name = response_message["function_call"]["name"]
            fn = fns_obj["reference_fn"][fn_name]
            fn_args = json.loads(response_message["function_call"]["arguments"])
            print(len(fn_args))
            messages.append(response_message),
            messages.append({
                "role": "function",
                "name": fn_name,
                "content": fn(request_type=fn_args.get("request_type"))
            })
            
            
            second_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages = messages,
                stream=True
            )
            
            return {
                "stream": True,
                "content": second_response
            }
    
    def stream_result(self, response, tts_cb_fn, listen_audio_cb_fn):
        sentence = ""
        full_response = ""
        pattern = r"[a-zA-Z][.?!]$"
        code_pattern = r"```[python|csharp].+?```"
        for chunk in response:
            try:
                chunk_text = chunk["choices"][0]["delta"]["content"]
                full_response += chunk_text
                sentence += chunk_text
                match = re.search(pattern, sentence)
                
                code_blocks = []
                
                if '```python' in sentence:
                    print('called opening format of python')
                    if '\n```\n' in sentence:
                        print('parsed a closing code format')
                        code_pattern_match = re.search(code_pattern, sentence, re.DOTALL)
                        parsed_string = code_pattern_match.group(0)
                        sentence = sentence.replace(parsed_string, "")
                        code_blocks.append(parsed_string)
                        tts_cb_fn("__open_blackboard__", listen_audio_cb_fn)
                        
                        with open("gui/message_transfer.txt", "w") as f:
                            f.write(parsed_string)
                            
                        
                        
                
                if match and "```python" not in sentence:
                    tts_cb_fn(sentence, listen_audio_cb_fn)
                    sentence = ""
            
            except:
                pass
                
        # Ensure to send the last sentence to queue
        if sentence:
            tts_cb_fn(sentence, listen_audio_cb_fn)
        
        return full_response
    
    def compound_messages(self):
        # System input
        messages = []
        for system_input in self.system_inputs:
            obj = {
                "role": "system", "content": system_input
            }
            
            messages.append(obj)
            
            # TODO: Format input under system input
        
        # User input
        for user_input in self.user_inputs:
            obj = {
                "role": "user", "content": user_input
            }
            
            messages.append(obj)
            
        return messages
    
    
    def compound_functions(self):
        pass
    
    
    def set_personality(self, text, override=False):
        if override:
            self.personality = text
        else:
            self.personality += text
        
    def set_name(self, name):
        self.name = name
        
    def set_system(self, text):
        self.system_inputs.append(text)
        