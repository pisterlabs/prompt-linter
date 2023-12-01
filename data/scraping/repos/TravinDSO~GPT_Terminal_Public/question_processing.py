import os
import gc
import gzip
import json
import math
import logging
import time
import tkinter as tk
import xml.etree.ElementTree as ET
from datetime import datetime
from box_sdk_gen.ccg_auth import CCGConfig,BoxCCGAuth
from box_sdk_gen.developer_token_auth import BoxDeveloperTokenAuth
from box_sdk_gen.client import BoxClient
from notion_client import Client as NotionClient
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import SeleniumURLLoader, CSVLoader, NotionDBLoader
from langchain.callbacks import get_openai_callback


# Process the question and return the answer
# Also perform the indexing of the documents if needed
def process_question(total_docs_var,max_tokens_var,query_temp,openai_status_var,doc_text,env_file,data_use, query, prompt_style, data_folder,reindex=False,chat_history=[]):

    query_temp = query_temp.get()

    max_tokens = int(float(max_tokens_var.get()))

    doc_text.insert(tk.END, "Using environment file: " + env_file + "\n")
    doc_text.insert(tk.END, "Using data folder: " + data_folder + "\n")
    doc_text.update()
    load_dotenv(env_file, override=True)

    # Load the OPENAI environment variables from the .env file depending on use_azure
    use_azure = os.getenv("USE_AZURE")
    if use_azure.lower() == "true":
        USE_AZURE = True
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_API_ENDPOINT")
        os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
        EMBEDDINGS_MODEL = os.getenv("AZURE_EMBEDDINGS_MODEL")
        AZURE_OPENAI_API_MODEL = os.getenv("AZURE_OPENAI_API_MODEL")
        OpenAIEmbeddings.deployment = os.getenv("AZURE_OPENAI_API_MODEL")
    else:
        USE_AZURE = False
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")
        OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL")

    # Load the NOTION environment variables from the .env file depending on use_notion
    use_notion = os.getenv("USE_NOTION")
    if use_notion.lower() == "true":
        USE_NOTION = True
        NOTION_TOKEN = os.getenv("NOTION_API_KEY")
        DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
    else:
        USE_NOTION = False

    # Load the BOX environment variables from the .env file depending on use_notion
    use_box = os.getenv("USE_BOX")
    if use_box.lower() == "true":
        USE_BOX = True
        BOX_TOKEN = os.getenv("BOX_TOKEN")
        BOX_FOLDER_ID = os.getenv("BOX_FOLDER_ID")
    else:
        USE_BOX = False

    # Text splitter for splitting the text into chunks
    class CustomTextSplitter(CharacterTextSplitter):
        def __init__(self, separators, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.separators = separators

        def split_text(self, text):
            import re
            chunks = []
            pattern = '|'.join(map(re.escape, self.separators))
            splits = re.split(pattern, text)
            return self._merge_splits(splits, self.separators[0])

    previous_logging_level = logging.getLogger().getEffectiveLevel()
    # Temporarily set the logging level to suppress warnings
    logging.getLogger().setLevel(logging.ERROR)  # Set to logging.ERROR to suppress warnings


    if USE_AZURE:
        llm = ChatOpenAI(max_tokens=max_tokens,deployment_id=AZURE_OPENAI_API_MODEL,temperature=query_temp,top_p=1,frequency_penalty=0,presence_penalty=0)
    else:
        llm = ChatOpenAI(max_tokens=max_tokens,model_name=OPENAI_API_MODEL,temperature=query_temp,top_p=1,frequency_penalty=0,presence_penalty=0)

    prompt_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. The System defines the personality and instructions to modify the response.
    System: {prompt_style}
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    doc_chain = load_qa_chain(llm, chain_type="stuff")

    logging.getLogger().setLevel(previous_logging_level)

    chain_path = os.path.join(data_folder, 'chain.json')
    docsearch_path = os.path.join(data_folder, 'docsearch')
    url_file = os.path.join(data_folder, 'urls.txt')
    wurl_file = os.path.join(data_folder, 'wurls.txt') # A URL that will be walked to find more URLs and content
    compressed_raw_text_file = os.path.join(data_folder, 'temporary_cached_data.gz')
    add_docsearch_file = os.path.join(data_folder, 'add_docsearch.json')


    if os.path.exists(compressed_raw_text_file):
        os.remove(compressed_raw_text_file)

    if not os.path.exists(url_file):
        with open(url_file, 'w') as f:
            f.write('http://travin.com/blank')

    # Index the documents if needed
    # Do this if the chain file doesn't exist or if reindex is True
    # Do not index if data_use is 0 (no data query)
    if (not os.path.exists(chain_path) or reindex) and data_use > 0:

        skipped_path = ""

        openai_status_var.set("Reindexing documents...")
        with gzip.open(compressed_raw_text_file, 'wt', encoding='utf-8') as f:
            for root, _, files in os.walk(data_folder):
                for file in files:
                    if file.endswith('.pdf'):
                        pdf_path = os.path.join(root, file)
                        doc_text.insert(tk.END, f"Parsing: {pdf_path}\n")
                        doc_text.update()
                        doc_text.see(tk.END)
                        reader = PdfReader(pdf_path)
                        for i, page in enumerate(reader.pages):
                            text = page.extract_text()
                            if text:
                                f.write(text)
                        # Release memory after processing each PDF
                        del reader
                        gc.collect()
                    elif file.endswith('.csv'):
                        csv_path = os.path.join(root, file)
                        doc_text.insert(tk.END, f"Parsing: {csv_path}\n")
                        doc_text.update()
                        doc_text.see(tk.END)
                        reader = CSVLoader(csv_path)
                        data = reader.load()
                        for i, row in enumerate(data):
                            if row:
                                f.write(row.page_content)
                        # Release memory after processing each csv
                        del reader
                        gc.collect()
                    elif file.endswith('.txt'):
                        txt_path = os.path.join(root, file)
                        doc_text.insert(tk.END, f"Parsing: {txt_path}\n")
                        doc_text.update()
                        doc_text.see(tk.END)
                        with open(txt_path, 'r', encoding='utf-8') as txt_file:
                            txt_text = txt_file.read()
                        f.write(txt_text)
                    elif file.endswith('.xml'):
                        xml_path = os.path.join(root, file)
                        doc_text.insert(tk.END, f"Parsing: {xml_path}\n")
                        doc_text.update()
                        doc_text.see(tk.END)
                        # Create a context for iteratively parsing the XML file
                        context = ET.iterparse(xml_path, events=('start', 'end'))
                        context = iter(context)
                        # Process the XML file chunk by chunk
                        for event, elem in context:
                            if event == 'end':
                                # Write the text content of the current element to the gz file
                                if elem.text:
                                    f.write(elem.text)
                                # Clean up the processed element to save memory
                                elem.clear()
                    else:
                        skipped_path = skipped_path + "--" + os.path.join(root, file) + "\n"

            if skipped_path:
                doc_text.insert(tk.END, f"Unsupported Files:\n{skipped_path}\n")
                doc_text.update()
                doc_text.see(tk.END)

            if url_file and os.path.exists(url_file):
                with open(url_file, 'r') as url_file_obj:
                    url_list = [line.strip() for line in url_file_obj]
                url_loader = SeleniumURLLoader(urls=url_list)
                url_loader.headless = True
                url_loader.continue_on_failure = True
                url_loader.arguments = ['--disable-gpu','--log-level=3']
                url_data = url_loader.load()
                for i, data in enumerate(url_data):
                    text = data.page_content
                    f.write(text)

            if USE_BOX:
                if os.getenv("BOX_DEVELOPER_TOKEN"):
                    box_auth: BoxDeveloperTokenAuth = BoxDeveloperTokenAuth(token=BOX_TOKEN)
                else:
                    if os.getenv("BOX_ENTERPRISE_ID"):
                        box_oauth_config = CCGConfig(
                            client_id=os.getenv("BOX_CLIENT_ID"),
                            client_secret=os.getenv("BOX_CLIENT_SECRET"),
                            enterprise_id=os.getenv("BOX_ENTERPRISE_ID")
                        )
                    else:
                        box_oauth_config = CCGConfig(
                            client_id=os.getenv("BOX_CLIENT_ID"),
                            client_secret=os.getenv("BOX_CLIENT_SECRET"),
                            user_id=os.getenv("BOX_USER_ID")
                        )
                    box_auth = BoxCCGAuth(config=box_oauth_config)
                
                box_client: BoxClient = BoxClient(auth=box_auth)
                for box_item in box_client.folders.get_folder_items(BOX_FOLDER_ID).entries:
                    
                    boxfile_ext = box_item.name.split('.')[-1]
                    if boxfile_ext in ['vtt', 'txt', 'boxnote']:
                        boxfile_is_readable = True
                    else:
                        boxfile_is_readable = False

                    if box_item.type == 'file' and boxfile_is_readable:
                        try:
                            box_file = box_client.downloads.download_file(box_item.id).read()
                            if boxfile_ext == 'boxnote':
                                boxfile_data = json.loads(box_file.decode('utf-8'))
                                # Get the lastEditTimestamp value
                                timestamp_in_millis = boxfile_data.get('lastEditTimestamp')
                                if timestamp_in_millis:
                                    timestamp_in_seconds = timestamp_in_millis / 1000
                                    boxfile_timestamp = datetime.fromtimestamp(timestamp_in_seconds).strftime('%Y-%m-%d %H:%M:%S')

                                boxfile_text = boxfile_data.get('atext', {}).get('text', '')
                                f.write("Note name:" + box_item.name + " Date of note:" + boxfile_timestamp + " Note:" + boxfile_text)
                            elif boxfile_ext in ['vtt', 'txt']:
                                boxfile_text = box_file.decode('utf-8')
                                f.write("File name:" + box_item.name + " File Text:" + boxfile_text)

                            doc_text.insert(tk.END, f"Loaded box file: {box_item.name}\n")
                            doc_text.update()
                            doc_text.see(tk.END)
                        except Exception as e:
                            doc_text.insert(tk.END, f"Failed to load box file {box_item.name}: {e}\n")
                            doc_text.update()
                            doc_text.see(tk.END)
                    time.sleep(1)  # Rate limit pause

            if USE_NOTION:
                notion_loader = NotionDBLoader(
                    integration_token=NOTION_TOKEN,
                    database_id=DATABASE_ID,
                    request_timeout_sec=10,  # optional, defaults to 10
                )
                try:
                    notion_page_summaries = notion_loader._retrieve_page_summaries()
                except Exception as e:
                    doc_text.insert(tk.END, f"Failed to load notion pages: {e}\n")
                    doc_text.update()
                    doc_text.see(tk.END)
                    openai_status_var.set("Failed to load notion pages: " + str(e))

                notion_metadata_client = NotionClient(auth=NOTION_TOKEN)

                for each_page in notion_page_summaries:
                    attempt = 0
                    while attempt < 2:
                        try:
                            # https://developers.notion.com/reference/block
                            page_blocks = notion_loader.load_page(each_page)
                            page_metadata = notion_metadata_client.pages.retrieve(each_page['id'])

                            page_content = page_blocks.page_content

                            # Get page text from the page blocks
                            page_name = page_blocks.metadata['name']
                            try:
                                page_due = page_metadata['properties']['Due']['date']
                            except:
                                page_due = None
                            try:
                                page_status = page_metadata['properties']['Status']['select']['name']
                            except:
                                page_status = None
                            try:
                                page_labels = page_metadata['properties']['Label']['multi_select'][0]['name']
                            except:
                                page_labels = None

                            # Write the page text to the gz file
                            write_str = ''
                            if page_name:
                                write_str += f"Page Title:{page_name}\n"
                            if page_due:
                                write_str += f"|Page Date Due:{page_due}\n"
                            if page_status:
                                write_str += f"|Page Status:{page_status}\n"
                            if page_labels:
                                write_str += f"|Page Labels:{page_labels}\n"
                            if page_content:
                                write_str += f"|Page Content:{page_content}\n"
                            f.write(write_str)

                            if attempt == 0:
                                doc_text.insert(tk.END, f"Loaded page: {page_name}\n")
                            else:
                                doc_text.insert(tk.END, f"Surccessfly loaded page: {page_name} after retry\n")
                            doc_text.update()
                            doc_text.see(tk.END)
                            break  # if successful, break out of the while loop
                        except Exception as e:
                            attempt += 1
                            doc_text.insert(tk.END, f"Attempt {attempt} failed to load page {each_page['id']} : {e}\n")
                            doc_text.update()
                            doc_text.see(tk.END)
                            if attempt >= 2:
                                #print(f"Failed to load page {page_id} after {attempt} attempts")
                                doc_text.insert(tk.END, f"Failed to load page {each_page['id']} after {attempt} attempts\n")
                                doc_text.update()
                                doc_text.see(tk.END)

    if (not os.path.exists(chain_path) or reindex) and data_use > 0:
        # Initialize an empty list to store processed text chunks
        processed_texts_cache = []

        #Need to replace the magic numbers with variables and include them in the environment file
        with gzip.open(compressed_raw_text_file, 'rt', encoding='utf-8') as f:
            text_splitter = CustomTextSplitter(
                separators=['\n', '. '],
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
            )
            
            current_chunk = ''
            for line in f:
                current_chunk += line
                if len(current_chunk) >= text_splitter._chunk_size:  # Corrected attribute name
                    # Process the current chunk
                    processed_chunk = text_splitter.split_text(current_chunk)
                    
                    # Append the processed chunk to the cache
                    processed_texts_cache.extend(processed_chunk)
                    
                    # Keep the chunk_overlap part of the current chunk for the next iteration
                    current_chunk = current_chunk[-text_splitter._chunk_overlap:]  # Corrected attribute name

        # Process the remaining part of the last chunk
        if current_chunk:
            processed_chunk = text_splitter.split_text(current_chunk)
            processed_texts_cache.extend(processed_chunk)

        os.remove(compressed_raw_text_file)

        if USE_AZURE:
            embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL,chunk_size=16)
        else:
            embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL,chunk_size=500)

        docsearch = FAISS.from_texts(processed_texts_cache, embeddings)
        docsearch.save_local(docsearch_path)
        doc_chain.save(chain_path)
    elif data_use > 0:
        if USE_AZURE:
            embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL,chunk_size=16)
        else:
            embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL,chunk_size=500)

        docsearch = FAISS.load_local(docsearch_path, embeddings)

    if data_use > 0:
        # Load additional docsearch instances and combine them
        if os.path.exists(add_docsearch_file):
            with open(add_docsearch_file, 'r') as f:
                add_docsearch_config = json.load(f)

            additional_folders = add_docsearch_config.get('additional_folders', [])
            for folder in additional_folders:
                additional_docsearch_path = os.path.join(folder, 'docsearch')
                if os.path.exists(additional_docsearch_path):
                    #print(f"Loading additional docsearch from {additional_docsearch_path}")
                    additional_docsearch = FAISS.load_local(additional_docsearch_path, embeddings)
                    docsearch.merge_from(additional_docsearch)
                else:
                    doc_text.insert(tk.END, "Additional docsearch path " + additional_docsearch_path + " does not exist" + "\n")
                    doc_text.update()
                    doc_text.see(tk.END)

    openai_status = ""

    if query != '':

        total_tokens = ""
        openai_status = ""
        answer = ""

        if prompt_style:
            question = f"'role': 'system', 'content':{prompt_style}\n'role': 'system', 'user'{query}"
        else:
            question = f"{query}"

        if data_use == 1:
            number_of_docs = int(float(total_docs_var.get()))
            docs = docsearch.similarity_search(query, k=number_of_docs)

            with get_openai_callback() as cb:
                try:
                    answer = doc_chain.run(input_documents=docs, question=question)
                except Exception as e:
                    if "maximum context length" in str(e):
                        try:
                             #Rate limit pause
                            time.sleep(5)
                            # Extract max_context_length
                            max_context_length = int(str(e).split("maximum context length is ")[1].split(" tokens.")[0])
                            # Extract num_tokens
                            num_tokens = int(str(e).split("you requested ")[1].split(" tokens")[0])
                            number_of_docs = calculate_num_docs(num_tokens, max_context_length)
                            docs = docsearch.similarity_search(query, k=number_of_docs)
                            answer = doc_chain.run(input_documents=docs, question=question)
                            openai_status += "Maximum tokens exceeded. Temporary reduced documents to " + str(number_of_docs) + " | "
                        except:
                            try:
                                #Rate limit pause
                                time.sleep(5)
                                adjusted_number_of_docs = float(total_docs_var.get()) * 0.5
                                number_of_docs = (int(adjusted_number_of_docs))           
                                docs = docsearch.similarity_search(query, k=number_of_docs)
                                answer = doc_chain.run(input_documents=docs, question=question)
                                openai_status += "Maximum tokens exceeded. Temporary reduced documents to " + str(number_of_docs) + " | "
                            except:
                                try:
                                    #Rate limit pause
                                    time.sleep(5)
                                    number_of_docs = 5
                                    docs = docsearch.similarity_search(query, k=number_of_docs)
                                    answer = doc_chain.run(input_documents=docs, question=question)
                                    openai_status += "Maximum tokens exceeded. Temporary reduced documents to 5. | "
                                except:
                                    doc_text.insert(tk.END, "Error: " + str(e) + "\n")
                                    doc_text.update()
                                    answer = ""
                                    openai_status += "Error: " + str(e) + " | "
                    
            total_tokens = cb.total_tokens
        elif data_use == 2:
            number_of_docs = int(float(total_docs_var.get()))
            docs = docsearch.similarity_search_with_score(query, k=number_of_docs)
            answer = ""
        else:
            # Initialize an empty lists to store processed text chunks
            docs = []
            with get_openai_callback() as cb:
                try:
                    answer = doc_chain.run(input_documents=docs, question=question)
                except Exception as e:
                    print(e)
                    answer = ""
                total_tokens = cb.total_tokens

        if total_tokens:
                openai_status += "Total tokens used: " + str(total_tokens)

        return answer, docs, openai_status
    else:
        return "", None, openai_status

def calculate_num_docs(num_tokens, max_context_length):
    num_docs = 1000
    ratio = max_context_length / num_tokens
    num_docs = math.floor(ratio * num_docs)
    num_docs = num_docs // 10 * 10  # round down to nearest 10
    num_docs = num_docs - 5 # subtract 5 to be safe
    return num_docs