#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import time

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def react_for_logs(query):
    manual_react = f"""
    
    Find and report which logs breach the compliance policy. Provide actionable insights to mitigate the risk of future breaches.
    System Log: 
    [2023-08-20 12:15:32] [info] [client 192.168.0.1] GET /index.html HTTP/1.1 200 5124

    Compliance Breaches:
    Successful page request- The log entry shows a successful GET request (GET /index.html) from client IP 192.168.0.1, which returned a 200 status code. This indicates that the client accessed the homepage successfully, and the page size was 5124 bytes.

    Actionable Insights:
    Regularly review and monitor server configurations to ensure all required files and directories exist and are accessible.
    Implement proper access controls and permissions to restrict unauthorized access to sensitive files or system resources.


    Find and report which logs breach the compliance policy. Provide actionable insights to mitigate the risk of future breaches.
    System Log: 
    [2023-08-20 12:17:45] [error] [client 192.168.0.2] File does not exist: /var/www/html/includes/config.php

    Compliance Breaches:
    Missing file- The log entry indicates an error where the client at IP 192.168.0.2 requested a file (/var/www/html/includes/config.php) that does not exist. This could be an indication of a misconfiguration or an attempt to access sensitive files.

    Actionable Insights:
    Review the server configuration to ensure that the file path is correct and that sensitive files are not accessible to the public.
    Monitor the IP address 192.168.0.2 for further suspicious activity or repeated attempts to access sensitive files.


    Find and report which logs breach the compliance policy. Provide actionable insights to mitigate the risk of future breaches.
    System Log:
    [2023-08-20 12:19:10] [info] [client 192.168.0.3] POST /login.php HTTP/1.1 302 0

    Compliance Breaches:
    The log entry shows an info-level log indicating a POST request (POST /login.php) from client IP 192.168.0.3, which resulted in a 302 status code. This suggests a form submission, likely for user authentication or login.

    Actionable Insights:
    No immediate action is required unless there are suspicious patterns associated with this IP address.

    Find and report which logs breach the compliance policy. Provide actionable insights to mitigate the risk of future breaches.
    System Log:
    {query}

    """

    return manual_react

def react_for_system_policies(query):

    manual_react = f"""
        List out all the possible system policies that are being violated.
        {query}
    """

    return manual_react
    


def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(react_for_logs(query))
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Interactive LLM for logs and security analysis with vectorstores')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
