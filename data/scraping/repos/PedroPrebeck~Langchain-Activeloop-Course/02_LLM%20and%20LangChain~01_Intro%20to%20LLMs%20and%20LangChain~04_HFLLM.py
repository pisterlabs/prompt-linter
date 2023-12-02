from dotenv import load_dotenv

load_dotenv()

from langchain import PromptTemplate

template = """Question: {question}

Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])

question = "What is the capital city of France?"

from langchain import HuggingFaceHub, LLMChain

hub_llm = HuggingFaceHub(
    repo_id="google/flan-t5-large", model_kwargs={"temperature": 0}
)

llm_chain = LLMChain(prompt=prompt, llm=hub_llm)

print(llm_chain.run(question))

multi_template = """Answer the following questions in order.

Questions:
{questions}

Answers:
"""
long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

llm_chain = LLMChain(prompt=long_prompt, llm=hub_llm)

qs_str = (
    "What is the capital city of France?\n"
    + "What is the largest mammal on Earth?\n"
    + "Which gas is most abundant in Earth's atmosphere?\n"
    + "What color is a ripe banana?\n"
)
print(llm_chain.run(qs_str))

multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""
long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

llm_chain = LLMChain(prompt=long_prompt, llm=hub_llm)

qs_str = (
    "What is the capital city of France?\n"
    + "What is the largest mammal on Earth?\n"
    + "Which gas is most abundant in Earth's atmosphere?\n"
    + "What color is a ripe banana?\n"
)
print(llm_chain.run(qs_str))