from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.config import get_settings
from src.prompts import PROJECT_TEMPLATE, ProjectParams
from src.parsers import get_project_parser, ProjectIdeas

_SETTINGS = get_settings()


class TemplateLLM:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=_SETTINGS.model, openai_api_key=_SETTINGS.openai_key
        )
        self.parser = get_project_parser()
        self.prompt_template = PromptTemplate(
            template=PROJECT_TEMPLATE,
            input_variables=["major", "n_examples", "language"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def generate(self, params: ProjectParams) -> ProjectIdeas:
        _input = self.prompt_template.format(**params.dict())
        output = self.llm.predict(_input)
        return self.parser.parse(output)

    def generate_and_save(self, params: ProjectParams, out_file: str):
        output_obj = self.generate(params)
        with open(out_file, "w") as f:
            f.write(output_obj.json(ensure_ascii=False))
