from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

class ScriptGenerator:
    def __init__(self, api_key, model_name="gpt-3.5-turbo"):
        """
        Initialize the YTShortsScriptGenerator with the OpenAI API key and model name.

        Args:
            api_key (str): Your OpenAI API key.
            model_name (str, optional): The name of the OpenAI model to use. Default is "gpt-3.5-turbo".
        """
        self.llm = OpenAI(model_name=model_name, openai_api_key=api_key)
        self.parser = PydanticOutputParser(pydantic_object=ytShortsScript)

    def generate_scripts(self, category, sample_videos, num_of_scripts=1):
        """
        Generate short video scripts based on the provided category and sample videos.

        Args:
            category (str): The category for the video scripts.
            sample_videos (str): A string containing sample videos and their descriptions.
            num_of_scripts (int, optional): The number of scripts to generate. Default is 1.

        Returns:
            dict: A dictionary containing the generated video script(s).
        """
        try:
            prompt = PromptTemplate(
                template="you are a very good script writer for a {category} youtube channel. here are some scripts which perform very well on youtube.\n {sample_videos} \n write only {num_of_scripts} more similar script of similar length for the same category.\nformat_instructions:{format_instructions}",
                input_variables=['category', 'sample_videos', 'num_of_scripts'],
                partial_variables={'format_instructions': self.parser.get_format_instructions()},
            )

            input_data = prompt.format_prompt(category=category, sample_videos=sample_videos, num_of_scripts=num_of_scripts)
            output_data = self.llm(input_data.to_string())

            scriptObj =  self.parser.parse(output_data)
            return scriptObj.script
        except Exception as e:
            print("Error generating scripts:", str(e))
            return None


class ytShortsScript(BaseModel):
    script: str = Field(description="Script of the short video dont include anything in the start or end of the script (no video number)")
