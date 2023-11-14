from langchain.prompts.prompt import PromptTemplate


class DummyPromptTemplate(PromptTemplate):
    def __init__(self):
        super().__init__(input_variables=[], template="dummy template")
