from __future__ import annotations

import os
import openai
import yaml

from abc import ABC

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

### File interfacing with chatGPT to summarise content in a directory.
### This content happens to be Monty Python scripts
class LLMAccessor(ABC):
    prompt_instructions = """ Summarise each provided screenplay script into 2 sentences. """
    text_prompt = "text: What is your actual text answer to the prompt? Do not provide any other parameters. Only " \
                  "respond with your verbatim text answer to the prompt."

    def __init__(self):

        try:
            configs = yaml.safe_load(open("LLM_config.yaml", "r"))
        except yaml.YAMLError:
            print("Error loading LLM configurations")
            exit(1)

        os.environ["OPENAI_API_KEY"] = configs["OPENAI_API_KEY"]
        openai.api_key = configs["OPENAI_API_KEY"]
        self._temperature = configs["temp"]
        self._model = configs["model"]
        self._prompt: str = "Give me a text file summarising some text."

    # temp getter
    @property
    def temperature(self) -> int:
        return self._temperature

    # temp setter
    @temperature.setter
    def temperature(self, temperature: int):
        self._temperature = temperature

    # model getter
    @property
    def model(self) -> str:
        return self._model

    # model setter
    @model.setter
    def model(self, m):
        self._model = m

    def generate_prompts(self, script) -> dict[str, str]:
        text_template = f"""
            Please give a 2 sentence summary for the following text:

            text: {script}

            Format the response as JSON wrapped in curly brackets. JSON should have the following keys:
            - summary
            - full_script
            
            The key, summary, should have your generated summary - the output. The key, full_script, should have the 
            input {script}
            """

        summary_schema = ResponseSchema(
            name="summary",
            description=""
        )

        script_schema = ResponseSchema(
            name="script",
            description=""
        )

        response_schemas = [
            summary_schema,
            script_schema
        ]

        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        prompt_template = ChatPromptTemplate.from_template(template=text_template)

        messages = prompt_template.format_messages(
            format_instructions=format_instructions
        )

        chat = ChatOpenAI(temperature=self.temperature)

        response = chat(messages)
        print("*****************THIS IS THE RESPONSE***********************")
        print(response)

        #print("This is get parameters output dict:")
        #output_dict = output_parser.parse(response.content)
        #print(output_dict)

        return response


if __name__ == "__main__":
    folder = "../training_data/data"
    Accessor = LLMAccessor()

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):

            ## Get script data for script
            fullyqualified_path = folder + "/" + filename
            _input: str = ""
            with open(fullyqualified_path, 'r') as f:
                for line in f.readlines():
                    _input += line
            f.close()

            f = open("output.txt", "a")
            f.write(str(Accessor.generate_prompts(_input)))
            f.close()
