from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chains import SimpleSequentialChain

if __name__ == '__main__':
    llm = OpenAI(temperature=0.9)
    # Don't try to use HF, as it doesn't support local embeddings
    # https://github.com/hwchase17/langchain/issues/4438
    # llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.9, "max_length": 128})
    template = "What is a good name for a company that makes {product}?"

    first_prompt = PromptTemplate.from_template(template)

    first_chain = LLMChain(llm=llm, prompt=first_prompt)

    template = "Write a catch phrase for the following company: {company_name}"

    second_prompt = PromptTemplate.from_template(template)

    second_chain = LLMChain(llm=llm, prompt=second_prompt)

    sss = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=False)

    print(sss.run("Gaming Mice"))
