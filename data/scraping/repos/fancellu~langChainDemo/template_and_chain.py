from langchain import PromptTemplate
# LLM Chain=Prompt+LLM
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI

if __name__ == '__main__':
    # Simple prompt to OpenAI
    template = "You are a naming consultant for new companies. What is a good name for a company that makes {product}?"

    prompt = PromptTemplate.from_template(template)

    llm = OpenAI(temperature=0.9)

    chain = LLMChain(llm=llm, prompt=prompt)

    print(chain.run("colorful socks"))

    print(chain.run("Digital pianos"))

    # More complex prompt to flan
    template2 = "You are a naming consultant for new companies. What is a good name for a {company} that makes {product}?"

    prompt2 = PromptTemplate.from_template((template2))

    llm2 = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.9, "max_length": 64})
    # https://huggingface.co/google/flan-t5-base

    chain2 = LLMChain(llm=llm2, prompt=prompt2)

    print(chain.run({"company": "AI Startup", "product": "large language models"}))
