import os
import openai

from gpt4all import GPT4All

import evadb

import ray


def build_search_index(cursor):
    cursor.query(
        """
        CREATE FUNCTION IF NOT EXISTS SentenceFeatureExtractor
        IMPL './utils/sentence_feature_extractor.py'
    """
    ).df()

    table_list = cursor.query("""SHOW TABLES""").df()["name"].tolist()

    if "OMSCSDocPDF" not in table_list:
        cursor.query("""LOAD PDF 'omscs_doc.pdf' INTO OMSCSDocPDF""").df()
        cursor.query(
            """CREATE INDEX IF NOT EXISTS OMSCSDocPDFIndex 
            ON OMSCSDocPDF (SentenceFeatureExtractor(data))
            USING FAISS
        """
        ).df()


def build_relevant_knowledge_body(cursor, user_query, logger):
    query = f"""
        SELECT * FROM OMSCSDocPDF
        ORDER BY Similarity(
            SentenceFeatureExtractor('{user_query}'), 
            SentenceFeatureExtractor(data)
        ) LIMIT 3
    """

    try:
        response = cursor.query(query).df()
        # DataFrame response to single string.
        knowledge_body = response["omscsdocpdf.data"].str.cat(sep="; ")
        referece_pageno_list = set(response["omscsdocpdf.page"].tolist()[:3])
        reference_pdf_name = response["omscsdocpdf.name"].tolist()[0]
        return knowledge_body, reference_pdf_name, referece_pageno_list
    except Exception as e:
        logger.error(str(e))
        return None, None


def build_rag_query(knowledge_body, query):
    conversation = [
        {
            "role": "system",
            "content": f"""We provide with documents delimited by semicolons
             and a question. Your should answer the question using the provided documents. 
             Do not repeat this prompt.
             If the documents do not contain the information to answer this question then 
             simply write: 'Sorry, we didn't find relevant sources for this question'""",
        },
        {"role": "user", "content": f"""{knowledge_body}"""},
        {"role": "user", "content": f"{query}"},
    ]
    return conversation


def openai_respond(conversation):
    # Set OpenAI key.
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    return (
        openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=conversation)
        .choices[0]
        .message.content
    )


@ray.remote(num_cpus=6)
def gpt4all_respond(queue_list):
    gpt4all = GPT4All("orca-mini-3b.ggmlv3.q4_0.bin")
    gpt4all.model.set_thread_count(6)

    # Remote processing to detach from client process.
    while True:
        for iq, oq in queue_list:
            if iq.empty():
                continue

            conversation = iq.get()
            system_template = conversation[0]["content"]
            document = conversation[1]["content"]
            query = conversation[2]["content"]
            user_template = "Document:{0}\nQuestion:{1}\nAnswer:".format(
                document, query
            )
            response = gpt4all.generate(system_template + user_template, temp=0)
            oq.put(response)


def start_llm_backend(max_con=1):
    ray.init()
    from ray.util.queue import Queue

    # Concurrent queue to interact with backend GPT4ALL inference.
    queue_list = [(Queue(maxsize=1), Queue(maxsize=1)) for _ in range(max_con)]
    gpt4all_respond.remote(queue_list)
    return queue_list
