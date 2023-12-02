import openai as ai
import os
from dotenv import load_dotenv
load_dotenv()


class DocCreator:
    def __init__(self, doc, path):
        if (path == None or path.strip() == ''):
            raise ValueError('path is required')
        self.path = path
        self.type = doc[0]
        self.raw = doc[1]
        print(self.type)
        self.key = os.getenv("OPEN_AI_KEY")
        ai.api_key = self.key
        self.prompt_raw = "You will be given a combination of all the python files in the directory. Create a readme file for the directory"
        self.prompt_formatted = "You will be given a combination of all the python files in the directory. Create a readme file for the directory. The python files have been formatted to only include the function definitions, class definitions, and comments"

    def create(self):
        create_string = ''
        if (self.type == 'raw'):
            create_string = self.create_raw()
        else:
            create_string = self.create_formatted()
        joined_path = os.path.join(self.path, 'README.md')
        with open(joined_path, 'w') as f:
            f.write(create_string)
        return "README.md created successfully"

    def create_raw(self):
        response = ai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.prompt_raw},
                {"role": "user", "content": self.raw},
            ]
        )

        return response['choices'][0]['message']['content']

    def create_formatted(self):
        response = ai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.prompt_formatted},
                {"role": "user", "content": self.raw},
            ]
        )

        return response['choices'][0]['message']['content']
