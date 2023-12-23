from find_similar import retriever_chroma
from llm_openai import get_openai_llm
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from langchain.globals import set_debug, set_verbose, warnings

# check https://colab.research.google.com/drive/1gyGZn_LZNrYXYXa-pltFExbptIe7DAPe?usp=sharing#scrollTo=LZEo26mw8e5k

clear = lambda: os.system("clear")

prompt_template = """Utilize the text provided in the document below to answer the following question: {question}. Ensure to reference specific sections of the text in your response. If the document does not contain sufficient information to answer the question, use your own knowledge to provide a well-informed answer. Structure your answer in a clear and concise manner, summarizing key points from the document as necessary. Here's the document text for reference: {information}."""

llm = get_openai_llm()

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever_chroma(),
    return_source_documents=False,
)


def chat():
    input_question = "test"
    clear()
    print("Welcome! \t Write /exit to exit.")
    while True:
        print("\n")
        input_question = input("Question: ")
        if input_question == "q" or input_question == "Q":
            print("Goodbye! \n")
            break

        llm_response = chain(input_question)
        print("\n" + llm_response["result"])


if __name__ == "__main__":
    set_debug(False)
    set_verbose(False)
    warnings.defaultaction = "ignore"
    chat()
