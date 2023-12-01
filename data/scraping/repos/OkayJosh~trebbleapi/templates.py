import os

from langchain import OpenAI

from dotenv import dotenv_values


# Load the environment variables from .env
env_vars = dotenv_values('.env')

class PromptTemplate:
    def __init__(self, brand, template):
        self.brand = brand
        self.template = template

    @property
    def prompt_context(self):
        return f"Brand name of {self.brand.name}, ' \
                         'we describe ourself as {self.brand.description}. ' \
                         'We have a product or service of {self.brand.product_description} ' \
                         'write a post for engagement"
                         # 'write a post of engagement {self._include_latest_news()}"

    @property
    def prompt_template(self):
        return f"format:: header: {self.template.header}, footer: {self.template.footer}, body: {self.template.body}," \
               f"context: {self.prompt_context}"


class OpenAIPromptEngine:
    def __init__(self, template: PromptTemplate):
        self.llm = OpenAI(temperature=0.9, openai_api_key=env_vars['OPENAI_KEY'])
        self.template = template

    @property
    def engine(self):
        return self.llm.predict

    def run(self):
        return self.engine(self.template.prompt_template)
        # llm.predict(prompt.format(product='Custom Software'))

    def stream(self):
        yield self.engine(self.template.prompt_template)
