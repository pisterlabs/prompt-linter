from prompt.classification import CLASSIFICATION_PROMPT
from langchain.chains import LLMChain
from llm.llm import llm
from langchain.prompts import PromptTemplate


class Classification:

    @classmethod
    def class_type(cls, question):
        prompt = PromptTemplate.from_template(CLASSIFICATION_PROMPT)
        chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        type = chain.run(question)
        return type
