import os
import textwrap
from typing import Optional
from urllib.request import pathname2url

from deep_translator import GoogleTranslator
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from openai.error import AuthenticationError

from scrapalot_prompts.prompt_template import ScrapalotSystemPromptTemplate
from .app_environment import translate_dst, translate_src, translate_docs, translate_q, ingest_target_source_chunks, args, openai_use, ingest_embeddings_model, gpu_is_enabled, \
    chromaDB_manager


def print_hyperlink(doc):
    page_link = doc.metadata['source']
    abs_path = os.path.abspath(page_link)
    file_url = pathname2url(abs_path)

    # This is the URL-encoded path, which can be used in a browser or link
    print(f'\033[32m[!]\033[0m URL: file://{file_url}')

    # This is the original path, which might contain characters not allowed in URLs (like spaces)
    print(f'\033[32m[!]\033[0m Path: {page_link}')


def print_document_chunk(doc):
    document_page = doc.page_content.replace('\n', ' ')
    if translate_docs:
        document_page = GoogleTranslator(source=translate_src, target=translate_dst).translate(document_page)
    wrapper = textwrap.TextWrapper(initial_indent='\033[37m', subsequent_indent='\033[37m', width=120)
    print(f"{wrapper.fill(document_page)}\033[0m\n")
    print(f'\033[94m"n" -> next, "q" -> quit: \033[0m')
    user_input = input()
    if user_input.lower() == 'q':
        exit(0)


prompt_template_instance = ScrapalotSystemPromptTemplate('scrapalot_prompts/prompt_system.json')


async def process_database_question(database_name, llm, collection_name: Optional[str], filter_document: bool, filter_document_name: Optional[str], prompt=prompt_template_instance):
    embeddings_kwargs = {'device': 'cuda'} if gpu_is_enabled else {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = OpenAIEmbeddings() if openai_use else HuggingFaceInstructEmbeddings(
        model_name=ingest_embeddings_model, model_kwargs=embeddings_kwargs, encode_kwargs=encode_kwargs
    )
    persist_dir = f"./db/{database_name}"

    db = Chroma(persist_directory=persist_dir,
                embedding_function=embeddings,
                collection_name=collection_name if collection_name else args.collection,
                client_settings=chromaDB_manager.get_chroma_setting(persist_dir),
                client=chromaDB_manager.get_client(collection_name))

    search_kwargs = {
        "k": ingest_target_source_chunks if ingest_target_source_chunks else args.ingest_target_source_chunks,
        "score_threshold": .5
    }

    if filter_document:
        search_kwargs["filter"] = {'source': {'$eq': os.path.join('.', 'source_documents', database_name, filter_document_name)}}

    retriever = db.as_retriever(search_kwargs=search_kwargs)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, condense_question_prompt=prompt.prompt_template, retriever=retriever, chain_type="stuff", return_source_documents=not args.hide_source)
    return qa


def process_query(qa: BaseRetrievalQA, query: str, chat_history, chromadb_get_only_relevant_docs: bool, translate_answer: bool):
    try:

        if chromadb_get_only_relevant_docs:
            docs = qa.retriever.get_relevant_documents(query)
            return None, docs

        if translate_q:
            query_en = GoogleTranslator(source=translate_dst, target=translate_src).translate(query)
            res = qa({"question": query_en, "chat_history": chat_history})
        else:
            res = qa({"question": query, "chat_history": chat_history})

        # Print the question
        print(f"\nQuestion: {query}\n")

        answer, docs = res['answer'], res['source_documents']
        # Translate answer if necessary
        if translate_answer:
            answer = GoogleTranslator(source=translate_src, target=translate_dst).translate(answer)

        print(f"\n\033[1m\033[97mAnswer: \"{answer}\"\033[0m\n")

        return answer, docs
    except AuthenticationError as e:
        print(f"Warning: Looks like your OPENAI_API_KEY is invalid: {e.error}")
        raise e
    except Exception as ex:
        print(f"Error: {ex}")
        raise ex
