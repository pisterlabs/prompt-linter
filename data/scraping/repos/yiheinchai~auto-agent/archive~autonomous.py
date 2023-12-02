"""
LLM conversations that generated this file:
https://sharegpt.com/c/yMsr1js (ChatGPT)
https://sharegpt.com/c/4U4Gbs6 (BingAI)
"""

import re

PROMPT = """
Welcome to the AutonomousLLM!
AutonomousLLM is a tool designed to improve itself by creating new methods to give it additional functionality. Its goal is to be able to self-activate, meaning that after the first initial input, it should not need additional inputs and will keep calling its own methods indefinitely.
AutonomousLLM is designed to be used through a command line interface. To interact with the AutonomousLLM, please use the following commands in your output:

1. To add a new method to the AutonomousLLM, use the following format:
   "add method METHOD_NAME: METHOD_CODE"
   
   This will add a new method with the specified name and code to the AutonomousLLM object.

2. To modify an existing method, use the following format:
   "modify method METHOD_NAME: METHOD_CODE"
   
   This will modify an existing method with the specified name and code in the AutonomousLLM object.

3. To execute a block of code, use the following format:
   "execute code CODE_TO_EXECUTE"
   
   This will execute the specified code using the AutonomousLLM object.

Keep in mind that all commands must be formatted correctly in your output to be executed properly by the AutonomousLLM.
"""

import openai


class AutonomousLLM:
    def __init__(self):
        self.api_key = "your_openai_api_key"
        openai.api_key = self.api_key

    def execute_code(self, output):
        add_method_pattern = r"add method (\w+) to class:\n([\s\S]+)"
        modify_method_pattern = r"modify method (\w+) in class:\n([\s\S]+)"
        execute_code_pattern = r"execute code:\n([\s\S]+)"

        try:
            add_method_match = re.match(add_method_pattern, output)
            modify_method_match = re.match(modify_method_pattern, output)
            execute_code_match = re.match(execute_code_pattern, output)

            if add_method_match:
                method_name, method_code = add_method_match.groups()
                setattr(self, method_name, method_code)
                print(f"Method {method_name} added to object.")
            elif modify_method_match:
                method_name, method_code = modify_method_match.groups()
                setattr(self, method_name, method_code)
                print(f"Method {method_name} modified in object.")
            elif execute_code_match:
                code = execute_code_match.group(1)
                exec(code)
                print("Code executed.")
            else:
                print("Invalid output.")
        except Exception as e:
            print(f"Error: {e}")

    def interface_chatgpt(self, input):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=PROMPT + input,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )
        output = response.choices[0].text.strip()
        return self.execute_code(output)

autonomous_llm = AutonomousLLM()
