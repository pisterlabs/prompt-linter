from langchain.chains.llm import LLMChain
from langchain.llms import OpenAI
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore
import os

def format_chat_history(chat_history_tuple: tuple) -> str:   
    chat_history = ""
    for dialogue_turn in chat_history_tuple:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        chat_history += "\n" + "\n".join([human, ai])
    
    return chat_history

def get_new_question_prompt(chat_history: str, user_question: str) -> str:
    
    new_question_template = """Given the following conversation and a follow up question, 
    rephrase the follow up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    
    new_question_prompt = PromptTemplate(template=new_question_template, input_variables=["chat_history", "question"])
    new_question_prompt = new_question_prompt.format_prompt(chat_history=chat_history, question = user_question).to_string()

    return new_question_prompt


def get_new_question(new_question_prompt: str) -> str:
    
    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True
        )
    
    question_generator = LLMChain.from_string(
        llm=question_gen_llm, template=new_question_prompt
    )

    new_question = question_generator.predict()

    return new_question


def get_context(new_question: str, vectorstore: VectorStore) -> str:

    context = vectorstore.similarity_search(new_question)
    chunks = []
    for i in context:
        chunks.append(i.page_content)
    return str(chunks)


def get_final_prompt(context: str, new_question: str) -> str:
    
    recruit_template  = """You are Hire Smith, a helpful AI assistant for job recruiters. 
    Given the following pieces of context from a single or multiple resumes, answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Do not mix data across seperate chunks and files which are stated as Filename. 

    {context}

    Question: {question}
    Helpful Answer:
    """
    final_prompt = PromptTemplate(template=recruit_template, input_variables=["context", "question"])
    final_prompt = final_prompt.format_prompt(context=context,question = new_question).to_string()

    return final_prompt


def get_final_answer(final_prompt: str) -> str:
    
    answer_llm = OpenAI(
        verbose=True,
        temperature=0
    )
    
    answer_chain = LLMChain.from_string(llm=answer_llm, template=final_prompt)

    answer = answer_chain.predict()
    return answer


if __name__=="__main__":
    
    os.environ["OPENAI_API_KEY"] = "" #OpenAI Key
    
    vectorstore = FAISS.load_local("ChatBot/faiss_index", OpenAIEmbeddings())
    chat_history_global = []

    try:
        while True:

            question = input()

            if chat_history_global != []:
                chat_history = format_chat_history(chat_history_global)
                new_question_prompt = get_new_question_prompt(chat_history=chat_history, user_question=question)
                new_question = get_new_question(new_question_prompt=new_question_prompt)

            else:
                new_question = question

            chunks = get_context(new_question=new_question, vectorstore=vectorstore)
            
            final_prompt = get_final_prompt(context=chunks, new_question=new_question)

            final_answer = get_final_answer(final_prompt=final_prompt)
            print(final_answer)

            chat_history_global.append((question, final_answer))

            if len(chat_history_global)>=5:
                chat_history_global=chat_history_global[1:]

    except KeyboardInterrupt:
        pass
    