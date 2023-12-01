from langchain.llms import ChatGLM
from langchain import PromptTemplate, LLMChain

template = """{question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])

endpoint_url = "http://192.168.100.20:8000"

llm = ChatGLM(
    endpoint_url = endpoint_url,
    max_token = 80000,
    history = [["你好"]],
    top_p = 0.9,
    model_kwargs = {"sample_model_args": False}
)

llm_chain = LLMChain(prompt=prompt,llm=llm)
question = "你是谁"
llm_chain.run(question)
