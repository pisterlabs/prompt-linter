from langchain import PromptTemplate
from langchain.llms import OpenAI
import os
from config import set_global_api_key
from pathlib import Path
from typing import List
from paths import REPO_DIR_PATH

set_global_api_key()


class SkillsToList:
    """
    This is a query engine which does extract skills from plain text skill descriptions 
    """
    def __init__(self, model_name: str, temperature: float, prompt_template_path: str):
        """
        :param model_name: Name of the LLM to be used for prompting
        :param temperature: Temperature prompt parameter, high - explorative, low - conservative
        :param prompt_template_path: Filepath to the prompt template
        """
        # initialize the LLM Api
        self.openai_engine = OpenAI(
            model_name=model_name,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            temperature=temperature
        )
        
        # Read a prompt template for skill extraction from a list of skills 
        prompt_path =  REPO_DIR_PATH / "esco_skill_graph" / prompt_template_path
        with prompt_path.open("r", encoding="utf-8") as f:
            template = f.read()

        # Create a prompt template 
        self.prompt_template = PromptTemplate(
            input_variables=["context"],
            template=template
        )

    def extract_skill_list(self, skill_description: str) -> List[str]:
        """"
        This method extracts a list of skills 

        :param skill_description: A Descriptive text outlining Soft-skills & Hard-Skills

        :return: List of skills 
        """
        prompt = self.prompt_template.format(context=skill_description)

        result = self.openai_engine(prompt)

        skills = [skill.strip() for skill in result.strip().split(";")]

        return skills 
    
    def __call__(self, skill_descriptions: List[str]) -> List[List[str]]:
        """
        This method returns lists of separate skills 

        :param skill_descriptions: List of plain text skill descriptions

        :return: Lists of skills 
        """
        skill_lists = []
        for skill_desc in skill_descriptions:
            skills = self.extract_skill_list(skill_description=skill_desc)
            skill_lists.append(skills)

        return skill_lists
