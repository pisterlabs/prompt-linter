from langchain import HuggingFaceHub, LLMChain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceHub(
    repo_id="gpt2",
    model_kwargs={
        'temperature': 0.3,
        'max_length': 200,
        'early_stopping': True,
        'no_repeat_ngram_size': 1,
    }
)

prompt = PromptTemplate(
    input_variables=["question"],
    template="Translate English to SQL: {question}",
)

chain = LLMChain(prompt=prompt, llm=llm, verbose=False)

print(f'---')
print(chain.run("what is the average age of customers"))
