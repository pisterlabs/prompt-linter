import os
from flask import Response
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

def run(file_path):
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')

    pdf_file = PyPDFLoader(file_path)
    pdf_data = pdf_file.load()

    pdf_text = ""

    for page in pdf_data:
        pdf_text += page.page_content

    questions_text_splitter = TokenTextSplitter(model_name="gpt-3.5-turbo-16k", chunk_size=10000, chunk_overlap=200)
    questions_text_chunks = questions_text_splitter.split_text(pdf_text)

    questions_documents = [Document(page_content=t) for t in questions_text_chunks]

    answers_text_splitter = TokenTextSplitter(model_name="gpt-3.5-turbo-16k", chunk_size=500, chunk_overlap=100)
    answers_documents = answers_text_splitter.split_documents(questions_documents)

    questions_llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k", temperature=0.5)

    prompt_template = """
    Take the role of an experienced creator of practice questions from study material. Aim to provide the five most valuable questions for an exam
    from the following text:
    {text}
    Questions:
    """
    questions_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_prompt_template = """
    Take the role of an experienced creator of practice questions from study material. We have the following questions: {existing_answer}
    Improve the five questions if you find in necessary and if not just provide the original questions. We are trying to study the most valuable
    questions from the following text:
    {text}
    Questions:
    """
    refine_questions_prompt = PromptTemplate(template=refine_prompt_template, input_variables=["existing_answer", "text"])

    questions_chain = load_summarize_chain(llm = questions_llm, chain_type="refine", verbose=True, question_prompt=questions_prompt, refine_prompt=refine_questions_prompt)

    questions = questions_chain(questions_documents)
    print(questions["output_text"], end='\n')

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma.from_documents(answers_documents, embeddings)

    answers_llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k", temperature=0.2)

    questions_list = questions["output_text"].split("/n")

    answers_chain = RetrievalQA.from_chain_type(llm = answers_llm, chain_type="stuff", verbose=True, retriever=vector_store.as_retriever(k=2))

    study_material = ""

    for question in questions_list:
        print("Question: ", question)
        answer = answers_chain.run(question)
        print("Answer: ", answer)
        print("-------------------------------\n")
        with open("./data/study.txt", "a") as file:
            file.write("Question: " + question + "\n")
            file.write("Answer: " + answer + "\n")
            file.write("-------------------------------\n")
            study_material += f"Question: {question}\nAnswer: {answer}\n-------------------------------\n"
            file.close()

    return Response(study_material, mimetype='text/plain')
