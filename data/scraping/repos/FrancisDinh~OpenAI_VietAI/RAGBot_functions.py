from langchain.docstore.document import Document
from utils import search_doc_from_knowledge_base, get_doc_pages
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import openai
import random
import ast


def create_qa_chain(**params):
    llm = OpenAI(**params)
    chain = load_qa_chain(llm, chain_type='stuff')
    return chain


def answer_idk():
    answers = [
        "I'm sorry, but I don't have that information.",
        "Unfortunately, I don't have the answer you're looking for.",
        "I'm not familiar with that topic, so I can't provide any insight.",
        "I'm afraid I don't know the answer to your question."
    ]
    return random.choice(answers)


def answer_idk_ext_context():
    answers = [
        "I'm sorry, but I don't have that information in your PDF. Do you want me to use my knowledge?",
        "Unfortunately, your PDF doesn't have the answer you're looking for. Do you want to search for it externally?",
        "I'm not familiar with that topic, so I can't provide any insight. Do you want me to use my knowledge?",
        "I'm afraid I don't know the answer to your question. Do you want me to use my knowledge?"
    ]
    return random.choice(answers)


def answer_what_else():
    answers = [
        "Sure, how else may I assist you today?",
        "Certainly, is there anything else you need help with right now?",
        "Of course, what other assistance do you require at the moment?",
        "Alright then, what else can I help you with today?"
    ]
    return random.choice(answers)


def answer_default(question, model_name, max_tokens, temperature):
    # Define parameters for the OpenAI ChatCompletion API
    params = dict(model=model_name,
                  temperature=temperature,
                  messages=[{"role": "user", "content": question}],
                  max_tokens=max_tokens,)

    # Generate a response from the AI model
    response = openai.ChatCompletion.create(**params)
    content = response.choices[0].message.content

    return content


def answer_from_doc(chain, doc, question: str):
    # Run the question-answering chain to find the answer to the question
    response = chain.run(
        input_documents=doc, question=question)

    return response


def what_is_the_question(question):
    question_template = f"""
    Here is the question user is asking {question}
    """
    params = dict(model="gpt-3.5-turbo",
                  messages=[{"role": "user", "content": question}])

    # Generate a response from the AI model
    response = openai.ChatCompletion.create(**params)
    content = response.choices[0].message.content

    return content


def is_yes_no(answer):
    question_full = f"""
    Does the answer contain a positive word, such as: yes, yeah, go ahead, approval? Answer only True or False 
    
    ###
    Answer: {answer}
    """
    return ast.literal_eval(answer_default(question_full, model_name='gpt-3.5-turbo', max_tokens=10, temperature=0))


def is_same_page(question):
    question_full = f"""
    Does the question contain 'the same page' or in the 'current context' or anything similar? Answer only True or False
    Example: 
    what is a loss function, using the same page => Return True
    Can you look for the answer to what is a neural network in page 1, 2 and also 3 => Return False
    
    ###
    Question: {question}
    """
    return ast.literal_eval(answer_default(question_full, model_name='gpt-3.5-turbo', max_tokens=10, temperature=0))


def has_page_numbers(question):
    question_full = f"""
    Does the below input contain page numbers? If yes, which page, answer as a list of integer. If not, return an empty list 
    
    ###
    Question: {question}
    """
    return ast.literal_eval(answer_default(question_full, model_name='gpt-3.5-turbo', max_tokens=100, temperature=0))


def extract_core_request(question):
    question_full = f"""
    what is the most important part of the request? Remove context such as: page number and same page.
    
    ###
    Request: {question}
    """
    return answer_default(question_full, model_name='gpt-3.5-turbo', max_tokens=100, temperature=0)


def is_request(text):
    question_full = f"""
    Is this input a request/question or not? Answer only True or False
    Example:
    What is neural network? => True
    Give me the definition of neural network => True
    Yes, have a nice day => False
    
    ###
    Input: {text}
    """
    return ast.literal_eval(answer_default(question_full, model_name='gpt-3.5-turbo', max_tokens=10, temperature=0))


def get_last_question(chat_history):
    # Return the last question when user agree to use external source
    question = chat_history[-3]["content"]
    return question


def get_pape_number_from_doc():
    pass
