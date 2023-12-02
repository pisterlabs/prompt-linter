from core.prompt import *
from core.pdf2txt import convert_pdf
import os
import openai
import sys
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv


sys.path.append("../..")
_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ["OPENAI_API_KEY"]

memory = ConversationBufferMemory(memory_key="chat_history", return_message=True)


def extract_tfq(question_list):
    question_list = question_list.strip().split("\n")
    questions = {"easy": [], "medium": [], "hard": []}
    current_difficulty = None
    current_question = {}

    for line in question_list:
        if line.startswith("Easy question"):
            current_difficulty = "easy"
            current_question = {}
        elif line.startswith("Medium question"):
            current_difficulty = "medium"
            current_question = {}
        elif line.startswith("Difficult question"):
            current_difficulty = "hard"
            current_question = {}
        elif line.startswith("Statement:"):
            current_question["question"] = line[len("Statement: ") :]
        elif line.startswith("Answer:"):
            current_question["answer"] = line[len("Answer: ") :]
        elif line.startswith("Explanation:"):
            current_question["explanation"] = line[len("Explanation: ") :]
            questions[current_difficulty].append(current_question)
    lists = []
    lists.extend(questions["easy"])
    lists.extend(questions["medium"])
    lists.extend(questions["hard"])
    return lists


def extract_mcq(question_list):
    questions = []

    current_question = {}

    for item in question_list.split("\n"):
        if "Easy question" in item:
            current_question["difficulty"] = "easy"
        elif "Medium question" in item:
            current_question["difficulty"] = "medium"
        elif "Difficult question" in item:
            current_question["difficulty"] = "hard"
        elif ": " in item:
            key, value = item.split(": ", 1)
            if key == "Question":
                current_question["question"] = value
                current_question["options"] = []
                current_question["true option"] = []
            elif key.startswith("Option"):
                current_question["options"].append(value)
            elif key == "True option":
                current_question["true option"].append(value)

                # When we have collected all information for the current question, add it to the list
                questions.append(current_question)
                current_question = {}

    return questions


def delete_file():
    import shutil

    chroma_dir = "chroma"
    for root, dirs, files in os.walk(chroma_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            os.remove(file_path)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            shutil.rmtree(dir_path)


def load_txt(path=""):
    # delete file
    delete_file()
    # load file
    from dotenv import load_dotenv, find_dotenv

    _ = load_dotenv(find_dotenv())
    from langchain.document_loaders import TextLoader

    loader = TextLoader(file_path=path, encoding="utf8")
    doc = []
    doc.extend(loader.load())
    persist_directory = "chroma"
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    vectordb.add_documents(documents=doc)


def load_file(file_path=""):
    x = file_path.split(".")
    if x[-1] == "pdf":
        convert_pdf(file_path)
        load_txt("./text.txt")
    else:
        load_txt(file_path)


def load_context(context=""):
    with open("./text.txt", "w", encoding="utf-8") as file:
        file.write(context)
        load_txt("./text.txt")


def genquests(l=None, type=None, e=None, m=None, h=None, question=None):
    template = ""
    if type == "tf":
        template = tfq_template
    elif type == "mcq":
        template = mcq_template
    elif type == "fill":
        template = fill_in_blank_template
    if question == "" or question == None:
        question = template.format(language=l, easy_num=e, med_num=m, hard_num=h)
    import datetime

    current_date = datetime.datetime.now().date()
    if current_date < datetime.date(2023, 9, 2):
        llm_name = "gpt-3.5-turbo-16k"
    else:
        llm_name = "gpt-3.5-turbo-16k"
    persist_directory = "server\chroma"
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=vectordb.as_retriever(search_kwargs={"k": 1}), memory=memory
    )
    result = qa_chain({"query": question})
    return result["result"]
