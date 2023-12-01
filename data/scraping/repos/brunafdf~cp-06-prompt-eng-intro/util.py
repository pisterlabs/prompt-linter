
"""
A Python module for interacting with OpenAI's GPT models and managing settings.

This module includes classes for managing settings and interacting with OpenAI's GPT models.

Classes:
- `Settings`: A class for managing settings using Pydantic's BaseSettings.
- `GPT_Wrapper`: A class for interfacing with OpenAI's GPT models.
"""
import os
from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field
from openai import OpenAI


# [i]                                                                                            #
# [i] Settings                                                                                   #
# [i]                                                                                            #

class Settings(BaseSettings):
    OPENAI_API_KEY: str = Field(validation_alias="OPENAI_API_KEY")


# [i]                                                                                            #
# [i] OpenAI API                                                                                 #
# [i]                                                                                            #

class GPT_Helper:
    def __init__(self, OPENAI_API_KEY):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    # TODO: Update GetCompletion method by using the get_completion from notebooks.
    def get_completion(self, prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]

        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0,
        )

        return completion.choices[0].message.content


# [i]                                                                                            #
# [i] Vars & Instances                                                                           #
# [i]                                                                                            #


"""
NOTE:

This code below is pivotal, especially in scenarios like `streamlit run`, where the Pydantic class BaseSettings might encounter difficulties loading environment variables.

The specific process within the code ensures the loading of these variables from the `.env` file.
This code segment employs the BaseSettings class from Pydantic to define settings, like the OPENAI_API_KEY.

The snippet also incorporates the loading of environment variables from the `.env` file, critical for scenarios where Pydantic may struggle with this process, ensuring smooth functionality, particularly in applications like streamlit run.
"""
_ = load_dotenv(find_dotenv())
if not _:
    _ = load_dotenv(".env")

print(os.getenv("OPENAI_API_KEY")[0:-15])

local_settings = Settings()
