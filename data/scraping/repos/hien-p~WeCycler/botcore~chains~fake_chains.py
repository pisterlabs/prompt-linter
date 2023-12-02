from langchain.llms.fake import FakeListLLM
from langchain import LLMChain, PromptTemplate

def build_fake_chain(response: str, num_responses: int) -> LLMChain:   
    responses=[response]*num_responses
    prompt_template = "Question: {question}\nAnswer in {language}:?"
    llm = FakeListLLM(responses=responses)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question", "language"])
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return llm_chain

def fake_qa_elec_chain(num_responses: int = 20) -> LLMChain:
    chain = build_fake_chain("electronic", num_responses)
    return chain

def fake_qa_recyc_chain(num_responses: int = 20) -> LLMChain:
    chain = build_fake_chain("recyclable", num_responses)
    return chain
