from langchain.prompts import StringPromptTemplate

human_prompt_template_v1 = """idea: {idea}"""
input_variables_v1 = ["idea"]

class HumanPromptTemplate(StringPromptTemplate):
    template: str

    def format(self, **kwargs):
        kwargs.update(self.partial_variables)
        return self.template.format(**kwargs)

human_prompt = HumanPromptTemplate(
    template = human_prompt_template_v1,
    input_variables = input_variables_v1
)