from langchain.llms import OpenAI

from demo.evaluators.oncall_action_evaluator.oncall_eval_chain import OncallEvalChain

def evalaute(error, command, index, _llm = OpenAI(temperature=0)):
    docs = index.similarity_search(error, k=1)
    inputs = [{"runbook": doc.page_content, "error": error, "command": command} for doc in docs]
    chain = OncallEvalChain.from_llm(_llm)
    print(chain.apply(inputs))