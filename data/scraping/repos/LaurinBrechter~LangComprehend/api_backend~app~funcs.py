import numpy as np
from pymongo.results import InsertOneResult
from pymongo import MongoClient
from langchain.document_loaders import YoutubeLoader
from langchain import PromptTemplate
from langchain.llms import OpenAI
import ast
import tiktoken
from typing import List
import datetime
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI


def cut_text(text, frac):
    splitted_text = text.split()
    n_words = len(splitted_text)
    # print(n_words)
    lim = int(frac*n_words)
    text_red = splitted_text[:lim]
    return " ".join(text_red), n_words


def get_video_text(url, language):
    loader = YoutubeLoader.from_youtube_url(url, language=language)
    result = loader.load()
    return result[0].page_content


def output_parser(text, llm):
    questions = ast.literal_eval(llm(
        f"""
        Please parse the following text in such a way that all the Questions are in one Python list.

        {text}
        
        """
    ))

    answers = ast.literal_eval(llm(
        f"""
        Please parse the following text in such a way that all the Answers are in one Python list. Only put the answers in the list.

        {text}
        
        """
    ))
    out_str = llm(
        f"""
        Please parse the following text in such a way that all the Topics are in one Python list. Only put the topics in the list.
        {text}
        """
    )
    print(out_str)
    topics = ast.literal_eval(out_str)

    return topics, questions, answers


def get_n_tokens(text) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def insert_query_to_db(
        client:MongoClient, 
        db_name:str, 
        collection_name:str, 
        text:str, 
        n_questions:int, 
        language:str, 
        url:str, 
        questions:list, 
        answers:list) -> InsertOneResult:
    
    document = {
    "text":text,
    "time_created":datetime.datetime.utcnow(),
    "n_questions":n_questions,
    "language_code":language,
    "url":url,
    "questions":questions,
    "answers":answers
    }
    
    db = client[db_name]
    collection = db[collection_name]

    res = collection.insert_one(document)

    return res


def add_vocab_to_db(client:MongoClient, db_name:str, collection_name:str, vocab:dict, u_id:int):
    docs = []
    
    for v in list(vocab.items()):
        docs.append({"u_id":u_id, "vocab":v[0], "inserted_at":datetime.datetime.utcnow(), "forms":v[1]})


    db = client[db_name]
    collection = db[collection_name]

    collection.insert_many(docs)



def get_response_chat(language, text):

    messages = [
        SystemMessage(content=f"""You are a helpful assistant that only provides answers in {language}"""),
        HumanMessage(content=text),
    ]

    return messages


def get_qa_topic(num_questions, text, language, fraction):
    prompt = PromptTemplate(
                            input_variables=["n_questions", "text", "language"],
                            template="""
                                - Can you come up with {n_questions} questions that test the comprehension that a user has for the following text delimited by triple backticks? 
                                Please also provide the 3 primary topics of the text.
                                ```{text}```. 
                                - Please provide the answers to the questions in {language}.
                                - Start each question with the following sign: 'Question: '.
                                - Start each answer with the following sign: 'Answer: '.
                                - Start the topics with the following sign:'Topics: '.
                                """
                                # - Delimit each question-answer pair with the following sign: '###' (three hashes).
                        )

    formatted = prompt.format(n_questions=num_questions, text=cut_text(text, frac=fraction), language=language)
    return formatted
                
    
def get_vocab(pipeline, text:str, irrel:list) -> dict:
    doc = pipeline(text)
    
    voc = {}
    ents = doc.ents
    irrel = ["PUNCT", "SPACE", "NUM"]

    for token in doc:
        tok_str = str(token).lower()
        lemma = token.lemma_.lower()
        if token.pos_ not in irrel:
            if lemma in voc.keys():
                if tok_str not in voc[lemma]:
                    voc[lemma].append(tok_str)
            else:
                voc[lemma] = [tok_str]
    # for vocab in voc.keys():
    #     print(voc[vocab], np.unique(voc[vocab]))
    #     voc[vocab] = list(set(voc[vocab]))

    return {"vocab":voc, "entities":ents}