from __future__ import annotations

from pathlib import Path
import os

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage    
)

from core.knowledgebase import constants


class TextAnalizer:
    def __init__(self: TextAnalizer) -> None: 

        self.model = ChatOpenAI(
            openai_api_key=constants.OPENAI_API_KEY,
            temperature=constants.LLM_MODEL_TEMPERATURE,
            model_name=constants.LLM_MODEL_NAME
        )

        self.prompt_names = [
            'prompt_generate', 'system_message_generate',
            'prompt_update', 'system_message_update',
            'prompt_question', 'system_message_question',
            'prompt_explain', 'system_message_explain', 
            'prompt_optimize', 'system_message_optimize', 
            'prompt_debug', 'system_message_debug',
        ]
        self.prompts = {}
        self.init_prompts()

        self.messages = []

        return
    
    def init_prompts(self: TextAnalizer) -> None:
        
        for prompt_name in self.prompt_names:
            prompt_path = Path(os.path.join(os.path.dirname(__file__), 'prompts', prompt_name))
            prompt_text = prompt_path.read_text()
            prompt_template = PromptTemplate.from_template(prompt_text)
            self.prompts[prompt_name] = prompt_template
        return


    def text_to_cypher_create(self: TextAnalizer, text: str, repo_path: str, file_path: str) -> str:
        self.messages = [
            SystemMessage(content=self.prompts['system_message_generate'].format()),
            HumanMessage(content=self.prompts['prompt_generate'].format(prompt=text, repo_path=repo_path, file_path=file_path))
        ]
        return self.model.predict_messages(self.messages).content


    def data_and_text_to_cypher_update(self: TextAnalizer, data: str, text: str, repo_path: str, file_path: str) -> str:
        self.messages = [
            SystemMessage(content=self.prompts['system_message_update'].format()),
            HumanMessage(content=self.prompts['prompt_update'].format(data=data, prompt=text, repo_path=repo_path, file_path=file_path))
        ]
        return self.model.predict_messages(self.messages).content
    
    def generate_questions(self: TextAnalizer, text: str) -> str:
        self.messages = [
            SystemMessage(content=self.prompts['system_message_question'].format()),
            HumanMessage(content=self.prompts['prompt_question'].format(prompt=text))
        ]
        return self.model.predict_messages(self.messages).content

    def _general_code_question(self: TextAnalizer, prompt_name: str, text: str) -> str:
        self.messages = [
            SystemMessage(content=self.prompts[f'system_message_{prompt_name}'].format()),
            HumanMessage(content=self.prompts[f'prompt_{prompt_name}'].format(code=text))
        ]
        return self.model.predict_messages(self.messages).content


    def optimize_code_style(self: TextAnalizer, text: str) -> str:
        return self._general_code_question('optimize', text)
    
    def explain_code(self: TextAnalizer, text: str) -> str:
        return self._general_code_question('explain', text)
    
    def debug_code(self: TextAnalizer, text: str) -> str:
        return self._general_code_question('debug', text)


if __name__ == '__main__':

    ta = TextAnalizer()

    example_reponame = 'History'
    example_repopath = os.path.join(os.path.dirname(__file__), 'examples', example_reponame)

    example_fname = 'napoleon.txt'
    example_fpath = os.path.join(example_repopath, example_fname)

    example_text = Path(example_fpath).read_text()
    
    ret = ta.text_to_cypher_create(example_text, example_repopath, example_fname)
    print(ret)

    questionable_text = """Napoleon initiated many liberal reforms that have persisted, 
    and is considered one of the greatest ever military commanders. His campaigns are still studied at military academies worldwide."""
    ret = ta.generate_questions(questionable_text)
    print(ret)

    code = """
        def evens(l):
            return [x for x in l if x % 2 == 0]
    """

    ret = ta.explain_code(code)
    print(ret)
