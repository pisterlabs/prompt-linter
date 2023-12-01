from Napari_Jupyter import Napari_Jupyter_coms
import openai
import re

class Napari_ChatGPT_coms(Napari_Jupyter_coms):
    def __init__(self):
        super().__init__()
        self.openaikey=None
    
    def set_open_ai_key(self,key):
        self.openaikey=key
        openai.api_key=self.openaikey
        
    def generate_code(self,operator,api_key):
        input_text = f"write a python function to take 8 bit image in numpy array format as input and perform {operator} on the array and return output image as numpy array from the function. Name the function as generated_custom_operation.Use opencv if possible.Provide only the code without explanations"
        prompt = f"\n{input_text}\n"
        completion = openai.ChatCompletion()
        # Use ChatGPT to generate code completion for the input text
        chat_log=[
            {"role": "system", "content": prompt}
        ]
        response = completion.create(model='gpt-3.5-turbo', messages=chat_log)
        code_block_with_message = response.choices[0]['message']['content']
        # Replace the original input text with the generated code block
        code_block = re.sub(r'^.*\n|\n.*$', '', code_block_with_message)
        return code_block


    def ask_anything(self,input_text):
        prompt = f"\n{input_text}\n"
        completion = openai.ChatCompletion()
        # Use ChatGPT to generate code completion for the input text
        chat_log=[
            {"role": "system", "content": prompt}
        ]
        response = completion.create(model='gpt-3.5-turbo', messages=chat_log)
        content = response.choices[0]['message']['content']
        return content

    
        