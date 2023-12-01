from langchain import PromptTemplate, HuggingFaceHub, LLMChain

template = """Question: {question}
        Answer: Let's think step by step."""

api_token = ''

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt,
                    llm=HuggingFaceHub(
                        repo_id="google/flan-t5-xl",
                        huggingfacehub_api_token=api_token,
                        model_kwargs={"temperature":0,"max_length":64}
                    ))

question = "What is the capital of France?"

print(llm_chain.run(question))