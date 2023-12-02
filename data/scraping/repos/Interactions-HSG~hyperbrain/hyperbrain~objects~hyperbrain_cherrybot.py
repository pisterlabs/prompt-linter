"""
2023-bt-raffael-rot
19-607-928
Prof. Dr. Simon Mayer
Danai Vachtsevanou, MSc.
"""

# Import utilities
import json
import requests
import datetime
import re
import time

API_URL = "https://api.openai.com/v1/chat/completions"  # Get the API URL of a model from ChatGPT


class HyperBrain:
    """
    """
    def __init__(self):
        """
        """
        with open('hyperbrain/data/API_KEY.txt', 'r') as f:
            API_KEY = f.read()  # GET API Key

        with open('hyperbrain/data/cherrybot_yaml.txt', 'r') as f:
            description_api = f.read()  # GET description of the API

        with open('memory.json', 'r') as f:
            memory = json.load(f)  # Get memory

        self._description = description_api  # Description of the API
        self._api_key_chat_gpt = API_KEY  # API key for the LLM
        self._memory = memory  # Memory of HyperBrain
        self._high_level_goal = str()  # High-level goal

    @staticmethod
    def _set_logs(log: str) -> int:
        """
        :param log: The log entry to save in the log file.
        :return: 0
        """
        now = datetime.datetime.now()  # Get the real time
        current_time = now.strftime("%H:%M:%S")  # Formatting of the date

        date_log = f"[{current_time}]  {log}\n"  # Append log entry to instance variable

        # Append a new line to the log file
        with open('hyperbrain/data/hyperbrain_cherrybot_logs.txt', 'a') as file:
            file.write(f"{date_log}")

        return 0

    @staticmethod
    def _get_python_code(data: str) -> str:
        """
        :param data: The response of the LLM from the action ask.
        :return: Extract and return all Python code from the given string.
        """
        code_string = ""

        python_code = re.findall('```(.*?)```', data, re.DOTALL)

        for i in range(len(python_code)):
            if "python" in python_code[i]:
                temp = python_code[i]
                python_code[i] = temp[6:]

        for item in python_code:
            code_string += item

        return code_string

    def _ask(self, query: str, model="gpt-4", temperature=0.9) -> str:
        """
        :param query: Query is the input for the LLM.
        :param model: Select the LLM model from OpenAI.
        :param temperature: Hyperparameter of the LLM to set the randomness.
        :return: Return the response of the LLM.
        """
        # Init the headers for the request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key_chat_gpt}"
        }

        # Init the data for the request
        data = {
            "model": model,
            "messages":
                [
                    {
                        "role": "system",
                        "content": "You are a helpful system to give instruction to interact with an API."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
            "temperature": temperature
        }

        time.sleep(3)
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))  # POST request to the OpenAi API

        data = response.json()  # Get the JSON data from the response

        result = data['choices'][0]['message']['content']  # Init the result of the request

        return result  # Return result

    def _thinking(self, action: str) -> int:
        """
        :param action: Instructions to interact with the robotic arm.
        :return: The method returns the status code from the request of the executed Python code.
        """

        query = f"Description '{self._description}'. Memory '{self._memory}'. Action '{action}'. " \
                f"Provide instructions in Python to process the action independently. " \
                f"Process the JSON file 'memory.json' with the results from the interaction." \


        self._set_logs(f"Memory: {self._memory}")

        answer = self._ask(query)

        code = self._get_python_code(answer)

        self._set_logs(f"Code: {code}")

        loc = {}

        time.sleep(10)

        exec(code, globals(), loc)

        self._set_logs(f"LOC: {loc}")

        status_code = loc['response'].status_code

        self._set_logs(status_code)

        with open('memory.json', 'r') as f:
            memory = json.load(f)  # Get memory
            self._memory = memory

        return status_code

    def hyperbrain(self) -> int:
        """

        """
        active = True

        while active:

            action = input("Please provide an action: ")

            self._high_level_goal = action

            executed = True

            while executed:

                status_code = self._thinking(self._high_level_goal)

                if status_code == 200:

                    if action == self._high_level_goal:
                        break
                    else:
                        action = self._high_level_goal

                elif status_code == 401:
                    action = "Register an operator"
                    status_code = self._thinking(action)

                elif status_code == 403:
                    action = "Get current operator"
                    status_code = self._thinking(action)

                elif status_code == 400:
                    status_code = self._thinking(action)

                print(f"Status code: {status_code}")

            if action == "x":
                break

        return 0


def main():
    pass


if __name__ == '__main__':
    main()
