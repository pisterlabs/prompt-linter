from langchain.docstore.document import Document
from utils import search_doc_from_knowledge_base, get_doc_pages
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import openai
import random


def create_qa_chain(**params):
    llm = OpenAI(**params)
    chain = load_qa_chain(llm, chain_type='stuff')
    return chain


def answer_idk():
    answers = [
        "I apologize, I dont have the answer for your question",
        "I am sorry, I dont have the answer in our pdf",
    ]
    return random.choice(answers)


def simple_ask_question(question):
    # Define parameters for the OpenAI ChatCompletion API
    params = dict(model="gpt-3.5-turbo",
                  messages=[{"role": "user", "content": question}])

    # Generate a response from the AI model
    response = openai.ChatCompletion.create(**params)
    content = response.choices[0].message.content

    return content


def answer_from_doc(chain, knowledge_base, question: str):
    doc = search_doc_from_knowledge_base(knowledge_base, question)

    if not doc:
        return answer_idk()
    # Run the question-answering chain to find the answer to the question
    response = chain.run(
        input_documents=doc, question=question)

    return response


def answer_retrieve_new_page(chain, knowledge_base, doc, question: str, retrieve_new_page=False):
    if retrieve_new_page:
        doc = search_doc_from_knowledge_base(knowledge_base, question)
        if not doc:
            return answer_idk()

    # Run the question-answering chain to find the answer to the question
    response = chain.run(
        input_documents=doc, question=question)

    return response


def answer_define_page(chain, question: str, pdf_reader, retrieve_page_id):
    # Assume: retrieve_page_id: List[Int]
    if not retrieve_page_id:
        return None
    else:
        docs = get_doc_pages(pdf_reader, retrieve_page_id)

    # Run the question-answering chain to find the answer to the question
    response = chain.run(
        input_documents=docs, question=question)

    return response


# want to trigger external search more
def answer_out_of_context(chain, knowledge_base, question: str, retrieve_external_knowledge=True, threshold=0.4):
    doc = search_doc_from_knowledge_base(
        knowledge_base, question, threshold=threshold)

    if doc is not None:
        # event if external flag is True, priority is on the original pdf
        response = chain.run(
            input_documents=doc, question=question)
    elif retrieve_external_knowledge == False and doc is None:
        response = "I dont know. Please enable retrieve_external_knowledge option and ask again."
    elif retrieve_external_knowledge == True and doc is None:
        # search out side
        response = simple_ask_question(question)
        response = "Reponse using external knowledge\n\n" + response
    return response
