#   Copyright [2023] [Sunholo ApS]
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import logging

from ..utils.parsers import remove_whitespace

from langchain.schema import Document
import langchain.text_splitter as text_splitter

def chunk_doc_to_docs(documents: list, extension: str = ".md", min_size: int = 800):
    """Turns a Document object into a list of many Document chunks.
       If a document or chunk is smaller than min_size, it will be merged with adjacent documents or chunks."""

    if documents is None:
        return None

    # Combine entire documents that are smaller than min_size
    combined_documents_content = ""
    combined_documents = []
    for document in documents:
        content = remove_whitespace(document.page_content)
        if len(content) < min_size:
            combined_documents_content += content + "\n"
            logging.info(f"Appending document as its smaller than {min_size}: length {len(content)}")
        else:
            if combined_documents_content:
                combined_documents.append(Document(page_content=combined_documents_content, metadata=document.metadata))
                combined_documents_content = ""
            combined_documents.append(document)

    if combined_documents_content:
        combined_documents.append(Document(page_content=combined_documents_content, metadata=documents[-1].metadata))

    source_chunks = []
    temporary_chunk = ""
    for document in combined_documents:
        splitter = choose_splitter(extension)
        for chunk in splitter.split_text(document.page_content):
            # If a chunk is smaller than the min_size, append it to temporary_chunk with a line break and continue
            if len(chunk) < min_size:
                temporary_chunk += chunk + "\n"
                logging.info(f"Appending chunk as its smaller than {min_size}: length {len(chunk)}")
                continue

            # If there's content in temporary_chunk, append it to the current chunk
            if temporary_chunk:
                chunk = temporary_chunk + chunk
                temporary_chunk = ""

            # If the combined chunk is still less than the min_size, append to temporary_chunk with a line break and continue
            if len(chunk) < min_size:
                temporary_chunk += chunk + "\n"
                logging.info(f"Appending chunk as its smaller than {min_size}: length {len(chunk)}")
                continue

            source_chunks.append(Document(page_content=chunk, metadata=document.metadata))

        # If there's any remaining content in temporary_chunk, append it as a new chunk
        if temporary_chunk:
            source_chunks.append(Document(page_content=temporary_chunk, metadata=document.metadata))
            temporary_chunk = ""

    # summarisation of large docs, send them in too
    #summaries = [Document(page_content="No summary made", metadata=metadata)]
    #do_summary = False #TODO: use metadata to determine a summary should be made
    #if documents is not None and do_summary:
    #    from summarise import summarise_docs
    #    summaries = summarise_docs(docs, vector_name=vector_name)
    #    summary_chunks = chunk_doc_to_docs(summaries)
    #    publish_chunks(summary_chunks, vector_name=vector_name)

    #    pubsub_manager = PubSubManager(vector_name, pubsub_topic=f"pubsub_state_messages")    
    #    pubsub_manager.publish_message(
    #        f"Sent doc chunks with metadata: {metadata} to {vector_name} embedding with summaries:\n{summaries}")

    logging.info(f"Chunked into {len(source_chunks)} documents")
    return source_chunks

def choose_splitter(extension: str, chunk_size: int=1024, chunk_overlap:int=0):
    if extension == ".py":
        return text_splitter.PythonCodeTextSplitter()
    elif extension == ".md":
        return text_splitter.MarkdownTextSplitter()
    
    return text_splitter.RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
