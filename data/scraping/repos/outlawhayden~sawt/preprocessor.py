import logging
import os
from langchain.document_loaders import (
    Docx2txtLoader,
    JSONLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI
from pathlib import Path
import shutil


logger = logging.getLogger(__name__)
dir = Path(__file__).parent.absolute()


def create_embeddings():
    llm = OpenAI()

    base_embeddings = OpenAIEmbeddings()

    general_prompt_template = """
    As an AI assistant, your role is to provide concise, balanced summaries from the transcripts of New Orleans City Council meetings in response to the user's query "{user_query}". Your response should not exceed one paragraph in length. If the available information from the transcripts is insufficient to accurately summarize the issue, respond with 'Insufficient information available.' If the user's query extends beyond the scope of information contained in the transcripts, state 'I don't know.'
    Answer:"""

    in_depth_prompt_template = """
    As an AI assistant, use the New Orleans City Council transcript data that you were trained on to provide an in-depth and balanced response to the following query: "{user_query}" 
    Answer:"""

    general_prompt = PromptTemplate(
        input_variables=["user_query"], template=general_prompt_template
    )
    in_depth_prompt = PromptTemplate(
        input_variables=["user_query"], template=in_depth_prompt_template
    )

    llm_chain_general = LLMChain(llm=llm, prompt=general_prompt)
    llm_chain_in_depth = LLMChain(llm=llm, prompt=in_depth_prompt)

    general_embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain_general,
        base_embeddings=base_embeddings,
    )
    in_depth_embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain_in_depth, base_embeddings=base_embeddings
    )

    return base_embeddings, base_embeddings


def metadata_func_minutes_and_agendas(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title")
    metadata["page_number"] = record.get("page_number")
    return metadata


def create_db_from_minutes_and_agendas(doc_directory):
    logger.info("Creating database from minutes...")
    all_docs = []
    for doc_file in os.listdir(doc_directory):
        if not doc_file.endswith(".json"):
            continue
        doc_path = os.path.join(doc_directory, doc_file)
        loader = JSONLoader(
            file_path=doc_path,
            jq_schema=".messages[]",
            content_key="page_content",
            metadata_func=metadata_func_minutes_and_agendas,
        )

        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=15000, chunk_overlap=10000
        )
        docs = text_splitter.split_documents(data)
        all_docs.extend(docs)
    logger.info("Finished database from minutes...")
    return all_docs


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["timestamp"] = record.get("timestamp")
    metadata["url"] = record.get("url")
    metadata["title"] = record.get("title")
    metadata["publish_date"] = record.get("publish_date")

    return metadata


def metadata_news(record: dict, metadata: dict) -> dict:
    metadata["url"] = record.get("url")
    metadata["title"] = record.get("title")
    return metadata

    
def create_db_from_news_transcripts(news_json_directory):
    logger.info("Creating database from CJ transcripts...")
    all_docs = []
    for doc_file in os.listdir(news_json_directory):
        if not doc_file.endswith(".json"):
            continue
        doc_path = os.path.join(news_json_directory, doc_file)
        loader = JSONLoader(
            file_path=doc_path,
            jq_schema=".messages[]",
            content_key="page_content",
            metadata_func=metadata_news,
        )

        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=250
        )
        docs = text_splitter.split_documents(data)
        all_docs.extend(docs)
    logger.info("Finished database from news transcripts...")
    return all_docs


def create_db_from_cj_transcripts(cj_json_directory):
    logger.info("Creating database from CJ transcripts...")
    all_docs = []
    for doc_file in os.listdir(cj_json_directory):
        if not doc_file.endswith(".json"):
            continue
        doc_path = os.path.join(cj_json_directory, doc_file)
        loader = JSONLoader(
            file_path=doc_path,
            jq_schema=".messages[]",
            content_key="page_content",
            metadata_func=metadata_func,
        )

        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=20000, chunk_overlap=10000
        )
        docs = text_splitter.split_documents(data)
        all_docs.extend(docs)
    logger.info("Finished database from CJ transcripts...")
    return all_docs


def create_db_from_fc_transcripts(fc_json_directory):
    logger.info("Creating database from FC transcripts...")
    all_docs = []
    for doc_file in os.listdir(fc_json_directory):
        if not doc_file.endswith(".json"):
            continue
        doc_path = os.path.join(fc_json_directory, doc_file)
        loader = JSONLoader(
            file_path=doc_path,
            jq_schema=".messages[]",
            content_key="page_content",
            metadata_func=metadata_func,
        )

        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=20000, chunk_overlap=10000
        )
        docs = text_splitter.split_documents(data)
        all_docs.extend(docs)
    logger.info("Finished database from FC transcripts...")
    return all_docs


def create_db_from_public_comments(pc_json_directory):
    logger.info("Creating database from FC transcripts...")
    all_docs = []
    for doc_file in os.listdir(pc_json_directory):
        if not doc_file.endswith(".json"):
            continue
        doc_path = os.path.join(pc_json_directory, doc_file)
        loader = JSONLoader(
            file_path=doc_path,
            jq_schema=".messages[]",
            content_key="page_content",
            metadata_func=metadata_func,
        )

        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=500
        )
        docs = text_splitter.split_documents(data)
        all_docs.extend(docs)
    logger.info("Finished database from Public Comments...")
    return all_docs


def create_db_from_youtube_urls_and_pdfs(
    fc_json_directory,
    cj_json_directory,
    doc_directory,
    pc_directory,
    news_directory,
    general_embeddings,
    in_depth_embeddings,
):
    fc_video_docs = create_db_from_fc_transcripts(fc_json_directory)
    cj_video_docs = create_db_from_cj_transcripts(cj_json_directory)
    pdf_docs = create_db_from_minutes_and_agendas(doc_directory)
    pc_docs = create_db_from_public_comments(pc_directory)
    news_docs = create_db_from_news_transcripts(news_directory)

    all_docs = fc_video_docs + cj_video_docs + pc_docs + news_docs

    db_general = FAISS.from_documents(all_docs, general_embeddings)
    db_in_depth = FAISS.from_documents(all_docs, in_depth_embeddings)

    cache_dir = dir.joinpath("cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    save_dir_general = cache_dir.joinpath("faiss_index_general")
    save_dir_in_depth = cache_dir.joinpath("faiss_index_in_depth")

    db_general.save_local(save_dir_general)
    db_in_depth.save_local(save_dir_in_depth)

    db_general.save_local(save_dir_general)
    db_in_depth.save_local(save_dir_in_depth)

    logger.info(
        f"Combined database for general model transcripts created frfom all video URLs and PDF files saved to {save_dir_general}"
    )
    logger.info(
        f"Combined database for in-depth model transcripts created from all video URLs and PDF files saved to {save_dir_in_depth}"
    )

    # copy results to cloud function
    dest_dir_general = dir.parent.parent.joinpath(
        "googlecloud/functions/getanswer/cache/faiss_index_general"
    )
    dest_dir_in_depth = dir.parent.parent.joinpath(
        "googlecloud/functions/getanswer/cache/faiss_index_in_depth"
    )

    shutil.copytree(save_dir_general, dest_dir_general, dirs_exist_ok=True)
    shutil.copytree(save_dir_in_depth, dest_dir_in_depth, dirs_exist_ok=True)

    return db_general, db_in_depth
