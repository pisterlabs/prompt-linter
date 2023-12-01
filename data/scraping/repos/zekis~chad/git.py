import traceback
import config
import json
import os
import re
import pika
import tempfile
import time 

import uuid
import subprocess
from typing import Optional
from bots.langchain_assistant import generate_response
#from common.card_factories import create_list_card, create_event_card
from common.utils import clean_and_tokenize, format_documents, format_user_question
#from common.utils import generate_response, generate_whatif_response, generate_plan_response
from common.rabbit_comms import publish

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.tools import BaseTool

from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader, NotebookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter


#load_dotenv(find_dotenv())

WHITE = "\033[37m"
GREEN = "\033[32m"
RESET_COLOR = "\033[0m"
model_name = "gpt-3.5-turbo"


def clone_github_repo(github_url, local_path):
    try:
        subprocess.run(['git', 'clone', github_url, local_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        return False

def load_and_index_files(repo_path):
    extensions = ['txt', 'md', 'markdown', 'rst', 'py', 'js', 'java', 'c', 'cpp', 'cs', 'go', 'rb', 'php', 'scala', 'html', 'htm', 'xml', 'json', 'yaml', 'yml', 'ini', 'toml', 'cfg', 'conf', 'sh', 'bash', 'css', 'scss', 'sql', 'gitignore', 'dockerignore', 'editorconfig', 'ipynb']

    file_type_counts = {}
    documents_dict = {}

    for ext in extensions:
        glob_pattern = f'**/*.{ext}'
        try:
            loader = None
            if ext == 'ipynb':
                loader = NotebookLoader(str(repo_path), include_outputs=True, max_output_length=20, remove_newline=True)
            else:
                loader = DirectoryLoader(repo_path, glob=glob_pattern)

            loaded_documents = loader.load() if callable(loader.load) else []
            if loaded_documents:
                file_type_counts[ext] = len(loaded_documents)
                for doc in loaded_documents:
                    file_path = doc.metadata['source']
                    relative_path = os.path.relpath(file_path, repo_path)
                    file_id = str(uuid.uuid4())
                    doc.metadata['source'] = relative_path
                    doc.metadata['file_id'] = file_id

                    documents_dict[file_id] = doc
        except Exception as e:
            print(f"Error loading files with pattern '{glob_pattern}': {e}")
            continue

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

    split_documents = []
    for file_id, original_doc in documents_dict.items():
        split_docs = text_splitter.split_documents([original_doc])
        for split_doc in split_docs:
            split_doc.metadata['file_id'] = original_doc.metadata['file_id']
            split_doc.metadata['source'] = original_doc.metadata['source']

        split_documents.extend(split_docs)

    index = None
    if split_documents:
        tokenized_documents = [clean_and_tokenize(doc.page_content) for doc in split_documents]
        index = BM25Okapi(tokenized_documents)
    return index, split_documents, file_type_counts, [doc.metadata['source'] for doc in split_documents]

def search_documents(query, index, documents, n_results=5):
    query_tokens = clean_and_tokenize(query)
    bm25_scores = index.get_scores(query_tokens)

    # Compute TF-IDF scores
    tfidf_vectorizer = TfidfVectorizer(tokenizer=clean_and_tokenize, lowercase=True, stop_words='english', use_idf=True, smooth_idf=True, sublinear_tf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform([doc.page_content for doc in documents])
    query_tfidf = tfidf_vectorizer.transform([query])

    # Compute Cosine Similarity scores
    cosine_sim_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Combine BM25 and Cosine Similarity scores
    combined_scores = bm25_scores * 0.5 + cosine_sim_scores * 0.5

    # Get unique top documents
    unique_top_document_indices = list(set(combined_scores.argsort()[::-1]))[:n_results]

    return [documents[i] for i in unique_top_document_indices]


def consume():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    message_channel = connection.channel()
    message_channel.queue_declare(queue='message')
    method, properties, body = message_channel.basic_get(queue='message', auto_ack=True)
    message_channel.close()
    
    return body



# def publish(message):
#     connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
#     message = encode_message('prompt', message)
#     notify_channel = connection.channel()
#     notify_channel.basic_publish(exchange='',
#                       routing_key='notify',
#                       body=message)
#     print(message)
#     notify_channel.close()

# def publish_action(message, button1, button2):
#     actions = [CardAction(
#         type=ActionTypes.im_back,
#         title=button1,
#         value=button1,
#     ),
#     CardAction(
#         type=ActionTypes.im_back,
#         title=button2,
#         value=button2,
#     )]
#     actions = [action.__dict__ for action in actions] if actions else []
#     message = encode_message('action', message, actions)

#     connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
#     notify_channel = connection.channel()
#     notify_channel.basic_publish(exchange='',
#                       routing_key='notify',
#                       body=message)
#     print(message)
#     notify_channel.close()


class git_review(BaseTool):
    name = "GIT_REVIEW"
    description = """useful for when you need to ask questions about a git repository
    To use the tool you must provide the following parameters "giturl", "query"
    Be careful to always use double quotes for strings in the json string 
    """

    #return_direct= True
    def _run(self, giturl: str, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        try:
            result = self.process_message(giturl, query)
            return result
            
        except Exception as e:
            traceback.print_exc()
            return f'To use the tool you must provide a valid giturl and query'
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("GIT_REVIEW does not support async")

    def process_message(self, github_url, query):
        repo_name = github_url.split("/")[-1]
        print("Cloning the repository...")
        local_path = "workspace/git"
        with tempfile.TemporaryDirectory() as local_path:
            if clone_github_repo(github_url, local_path):
                index, documents, file_type_counts, filenames = load_and_index_files(local_path)
                if index is None:
                    print("No documents were found to index. Exiting.")
                    publish("No documents were found to index. Exiting.")
                    return "No documents were found to index. Exiting."

                print("Repository cloned. Indexing files...")
                llm = OpenAI(api_key=config.OPENAI_API_KEY, temperature=0.2)

                template = """
                Repo: {repo_name} ({github_url}) | Conv: {conversation_history} | Docs: {numbered_documents} | Q: {question} | FileCount: {file_type_counts} | FileNames: {filenames}

                Instr:
                1. Answer based on context/docs.
                2. Focus on repo/code.
                3. Consider:
                    a. Purpose/features - describe.
                    b. Functions/code - provide details/samples.
                    c. Setup/usage - give instructions.
                4. Unsure? Say "I am not sure".

                Answer:
                """

                prompt = PromptTemplate(
                    template=template,
                    input_variables=["repo_name", "github_url", "conversation_history", "question", "numbered_documents", "file_type_counts", "filenames"]
                )

                llm_chain = LLMChain(prompt=prompt, llm=llm)

                conversation_history = ""
                question_context = QuestionContext(index, documents, llm_chain, model_name, repo_name, github_url, conversation_history, file_type_counts, filenames)
                                
                user_question = format_user_question(query)
                answer = ask_question(user_question, question_context)
                
                publish(answer)

                conversation_history += f"Question: {query}\nAnswer: {answer}\n"
                while True:
                    msg = consume()
                    if msg:
                        question = msg.decode("utf-8")
                        if question.lower() == "continue":
                            return "Done"
                        user_question = format_user_question(question)
                        answer = ask_question(user_question, question_context)
                        publish(answer)
                    time.sleep(0.5)
                return "Done"

            else:
                print("Failed to clone the repository.")
                publish("Failed to clone the repository.")
                return "Failed to clone the repository."



class QuestionContext:
    def __init__(self, index, documents, llm_chain, model_name, repo_name, github_url, conversation_history, file_type_counts, filenames):
        self.index = index
        self.documents = documents
        self.llm_chain = llm_chain
        self.model_name = model_name
        self.repo_name = repo_name
        self.github_url = github_url
        self.conversation_history = conversation_history
        self.file_type_counts = file_type_counts
        self.filenames = filenames

def ask_question(question, context: QuestionContext):
    relevant_docs = search_documents(question, context.index, context.documents, n_results=5)

    numbered_documents = format_documents(relevant_docs)
    question_context = f"This question is about the GitHub repository '{context.repo_name}' available at {context.github_url}. The most relevant documents are:\n\n{numbered_documents}"

    answer_with_sources = context.llm_chain.run(
        model=context.model_name,
        question=question,
        context=question_context,
        repo_name=context.repo_name,
        github_url=context.github_url,
        conversation_history=context.conversation_history,
        numbered_documents=numbered_documents,
        file_type_counts=context.file_type_counts,
        filenames=context.filenames
    )
    return answer_with_sources



