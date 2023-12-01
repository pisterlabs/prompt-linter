from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback


class KnowledgeIntegrator:
    def __init__(self, model_name, temperature):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    def get_answer(self, prompt, params):
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(prompt)
        )
        with get_openai_callback() as cb:
            result = llm_chain(params)
            costs = cb
        return result, {'total_cost': costs.total_cost}
