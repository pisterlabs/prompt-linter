import hashlib
import os
import uuid
from typing import List
import pinecone
import openai
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.docstore.document import Document


MODEL = "text-embedding-ada-002"


# API keys
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['LANGCHAIN_HANDLER'] = 'langchain'

os.environ['PINECONE_API_KEY'] = "b228e7a2-027a-4b3e-b65b-c06e8931c4e4"
os.environ['PINECONE_API_ENV'] = "gcp-starter"
os.environ['PINECONE_INDEX_NAME'] = "rag"


openai.api_key=os.environ['OPENAI_API_KEY']
pinecone.init( api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_API_ENV'])
pinecone_index=pinecone.Index(os.environ['PINECONE_INDEX_NAME'])


def upload_to_pinecone(text_document: str, file_name, chunk_size: int = 1000 ) -> None:
    """
    Upload the text content to pinecone

    @params
    text_document: text content needs to upload
    file_name: name of the filed to be included as metadata
    chunk_size: chunk size to split the data

    @return
    None
    """

    MODEL = "text-embedding-ada-002"

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = chunk_size,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    # text splitter
    texts = text_splitter.create_documents([text_document])

    for index, sub_docs in enumerate(texts):
        document_hash = hashlib.md5(sub_docs.page_content.encode("utf-8"))
        embedding = openai.embeddings.create(model= MODEL,input=sub_docs.page_content).data[0].embedding
        metadata = {"doc_name":file_name, "chunk": str(uuid.uuid4()), "text": sub_docs.page_content, "doc_index":index}
        pinecone_index.upsert([(document_hash.hexdigest(), embedding, metadata)])
        print("{} ==> Done".format(index))

    print("Done!")

    return True


def filter_matching_docs(question: str, top_chunks: int = 3, get_text: bool = False) -> List:
    """
    Semnatic search between user content and vector DB

    @param
    question: user question
    top_chunks: number of most similar content ot be filtered
    get_text: if True, return only the text not the document

    @return
    list of similar content
    """
    
    index=pinecone.Index(os.environ['PINECONE_INDEX_NAME'])

    question_embed_call = openai.embeddings.create(input = question ,model = MODEL)
    query_embeds = question_embed_call.data[0].embedding
    response = index.query(query_embeds,top_k = top_chunks,include_metadata = True)

    #get the data out
    filtered_data = []
    filtered_text = []

    for content in response["matches"]:
        #save the meta data as a dictionary
        info = {}
        info["id"] = content.get("id", "")
        info["score"] = content.get("score", "")
        # get the saved metadat info
        content_metadata = content.get("metadata","")
        # combine it it info
        info["filename"] = content_metadata.get("doc_name", "")
        info["chunk"] = content_metadata.get("chunk", "")
        info["text"] = content_metadata.get("text", "")
        filtered_text.append(content_metadata.get("text", ""))

        #append the data
        filtered_data.append(info)

    if get_text:
        similar_text = " ".join([text for text in filtered_text])
        print(similar_text)
        return similar_text

    print(filtered_data)

    return filtered_data


def QA_with_your_docs(user_question: str, text_list: List[str], chain_type: str = "stuff") -> str:
    """
    This is the main function to chat with the content you have

    @param
    user_question: question or context user wants to figure out
    text: list of similar texts
    chat_type: Type of chain run (stuff is cost effective)

    @return
    answers from the LLM
    """
    llm = OpenAI(temperature=0, openai_api_key = os.environ['OPENAI_API_KEY'])
    chain = load_qa_with_sources_chain(llm, chain_type = chain_type, verbose = False)

    all_docs = []
    for doc_content in text_list:
        metadata = {}
        doc_text = doc_content.get("text", "")
        metadata["id"] = doc_content.get("id", "")
        metadata["score"] = doc_content.get("score", "")
        metadata["filename"] = doc_content.get("filename", "")
        metadata["chunk"] = doc_content.get("chunk", "")
        chunk_name = doc_content.get("filename", "UNKNOWN")
        offset=", OFFSET="+str(doc_content.get("chunk","UNKNOWN"))
        metadata["source"] = chunk_name + offset
        all_docs.append(Document(page_content = doc_text, metadata = metadata))

    chain_response = chain.run(input_documents = all_docs, question = user_question )
    print(chain_response)

    return chain_response