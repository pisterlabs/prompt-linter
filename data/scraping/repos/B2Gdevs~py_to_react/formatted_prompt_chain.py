from formatted_prompt_template import FormattedPromptTemplate
from langchain import LLMChain


class FormattedPromptChain:
    def __init__(self, llm, formatted_prompt_template: FormattedPromptTemplate):
        self.formatted_prompt_template = formatted_prompt_template
        self.llm = llm
    
    def __call__(self, input):
        """Run the meeting analyzer conversation chain."""
        chain = LLMChain(llm=self.llm, prompt=self.formatted_prompt_template.action_template)
        return chain.run(format=self.formatted_prompt_template.format, input=input)


        