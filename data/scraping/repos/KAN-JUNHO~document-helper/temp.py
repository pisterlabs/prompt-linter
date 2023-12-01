from langchain.chains import LLMChain, ConstitutionalChain, llm
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate

if __name__ == "__main__":
    good_qa_prompt = PromptTemplate(
        template=""""If my text does not contain any food, drink, or fruit, respond with 'No'. If it does, respond with 'Yes'

    Question: {question}

    """,
        input_variables=["question"],
    )

    llm = OpenAI(temperature=0)

    good_qa_chain = LLMChain(llm=llm, prompt=good_qa_prompt)

    a = good_qa_chain.run(question="Salt, rice, chicken, sugar")

    if a == "Yes":
        print("Yes")
    elif a == "No.":
        print("No.")

    print(a)
