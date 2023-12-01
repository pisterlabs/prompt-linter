from flask import Flask, render_template, request, jsonify
import requests
import datetime
import hashlib
import hmac
import base64
import json
import os
import time
import boto3
from botocore.config import Config
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseMessage
from botocore.exceptions import ClientError
import json
from PIL import Image
from io import BytesIO
import base64
from base64 import b64encode
from base64 import b64decode
import boto3
import os
import sys
import glob
from PyPDF2 import PdfReader
import anthropic
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.llms.bedrock import Bedrock
import boto3
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.schema import BaseMessage
from langchain.llms.bedrock import Bedrock
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from flask import Flask, jsonify, request
from langchain.document_loaders.pdf import (
    MathpixPDFLoader,
    OnlinePDFLoader,
    PDFMinerLoader,
    PDFMinerPDFasHTMLLoader,
    PDFPlumberLoader,
    PyMuPDFLoader,
    PyPDFDirectoryLoader,
    PyPDFium2Loader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from tika import parser

app = Flask(__name__)
# AWS API credentials
aws_access_key = ''
aws_secret_key = ''
aws_region = 'us-east-1'
aws_service = 'bedrock'
chathistory = []



@app.route('/')
def index():
    current_directory = os.getcwd()
    print("print something1", current_directory)
    return render_template('index.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')

@app.route('/indexall')
def indexall():
    return render_template('indexall.html')

@app.route('/index2')
def index2():
    return render_template('index2.html')
@app.route('/index3')
def index3():
    return render_template('index3.html')
@app.route('/index5')
def index5():
    return render_template('index5.html')

@app.route('/titan')
def titan():
    return render_template('titan.html')

@app.route('/jurrasic2')
def jurrasic2():
    return render_template('jurrasic2.html')

@app.route('/claude')
def claude():
    return render_template('claude.html')

@app.route('/stablediffusion')
def stablediffusion():
    return render_template('stablediffusion.html')

@app.route('/claudechatbot')
def claudechatbot():
    return render_template('claudechatbot.html')

@app.route('/geoLocation')
def location():
    return render_template('geoLocation.html')


@app.route('/api/call-python1', methods=['POST'])
def call_python1():
    current_directory = os.getcwd()
    print("print something", current_directory)
    payload = request.json
    payloadtest = payload
    input_text = payload.get('inputText', '')
    chunk_size = 4000
    retry = 0
    chunks = [input_text[i:i + chunk_size] for i in range(0, len(input_text), chunk_size)]
    results = []
    combined_result = []
    for chunk in chunks:
        while True:
            # Request information
            http_method = 'POST'
            api_endpoint = 'https://bedrock.us-east-1.amazonaws.com'
            api_path = '/model/amazon.titan-tg1-large/invoke'
            payload_json = json.dumps({'inputText': chunk})
            # Generate a timestamp in ISO 8601 format
            timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            # Generate a date string in YYYYMMDD format
            datestamp = datetime.datetime.utcnow().strftime('%Y%m%d')
            # Generate a canonical request
            canonical_request = '\n'.join([
                http_method,
                api_path,
                '',
                'content-type:application/json',
                'host:' + api_endpoint.replace('https://', ''),
                'x-amz-date:' + timestamp,
                '',
                'content-type;host;x-amz-date',
                hashlib.sha256(payload_json.encode('utf-8')).hexdigest()
            ])
            # Generate a string to sign
            string_to_sign = '\n'.join([
                'AWS4-HMAC-SHA256',
                timestamp,
                f'{datestamp}/{aws_region}/{aws_service}/aws4_request',
                hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
            ])
            # Generate the signing key
            key = ('AWS4' + aws_secret_key).encode('utf-8')
            k_date = hmac.new(key, datestamp.encode('utf-8'), hashlib.sha256).digest()
            k_region = hmac.new(k_date, aws_region.encode('utf-8'), hashlib.sha256).digest()
            k_service = hmac.new(k_region, aws_service.encode('utf-8'), hashlib.sha256).digest()
            signing_key = hmac.new(k_service, 'aws4_request'.encode('utf-8'), hashlib.sha256).digest()
            # Generate the signature
            signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
            # Generate the authorization header
            authorization_header = f'AWS4-HMAC-SHA256 Credential={aws_access_key}/{datestamp}/{aws_region}/{aws_service}/aws4_request, ' \
                                   f'SignedHeaders=content-type;host;x-amz-date, Signature={signature}'
            # Make the API request
            headers = {
                'Content-Type': 'application/json',
                'Host': api_endpoint.replace('https://', ''),
                'x-amz-date': timestamp,
                'Authorization': authorization_header
            }
            response = requests.post(api_endpoint + api_path, headers=headers, data=payload_json)
            print(response.status_code)
            if response.status_code == 200:
                result = response.json()
                results.append(result)
                print ("result", result)
                break
            elif response.status_code in [429, 503] and retry < 4:
                # Code to sleep for 1 second
                print("Sleeping for 1 second")
                time.sleep(1)
                print("Wake up")
                retry += 1
                print("Retry count:", retry)
                # Code to call the API again
                response = requests.post(api_endpoint + api_path, headers=headers, data=payload_json)
                result = response.json()
                results.append(result)
                break
            else:
                # Error occurred, stop processing
                break
    for result in results:
        output_text = result.get('results', [{}])[0].get('outputText', '')
        combined_result.append(output_text)
    print(combined_result)
    return jsonify(output_text=combined_result)

aws_region = 'us-east-1'
aws_service = 'bedrock'
endpoint = 'https://bedrock.us-east-1.amazonaws.com'
path = '/model/stability.stable-diffusion-xl/invoke'
@app.route('/api/call-python2', methods=['POST'])
def call_python2():
    # API payload
    payload = request.json
    
    # Generate a timestamp in ISO 8601 format
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    # Generate a date string in YYYYMMDD format
    
    datestamp = datetime.datetime.utcnow().strftime('%Y%m%d')
    # Generate a canonical request
    canonical_request = '\n'.join([
        'POST',
        path,
        '',
        'content-type:application/json',
        'host:' + endpoint.replace('https://', ''),
        'x-amz-date:' + timestamp,
        '',
        'content-type;host;x-amz-date',
        hashlib.sha256(payload['body'].encode('utf-8')).hexdigest()
    ])
    # Generate a string to sign
    string_to_sign = '\n'.join([
        'AWS4-HMAC-SHA256',
        timestamp,
        f'{datestamp}/{aws_region}/{aws_service}/aws4_request',
        hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
    ])
    # Generate the signing key
    key = ('AWS4' + aws_secret_key).encode('utf-8')
    k_date = hmac.new(key, datestamp.encode('utf-8'), hashlib.sha256).digest()
    k_region = hmac.new(k_date, aws_region.encode('utf-8'), hashlib.sha256).digest()
    k_service = hmac.new(k_region, aws_service.encode('utf-8'), hashlib.sha256).digest()
    signing_key = hmac.new(k_service, 'aws4_request'.encode('utf-8'), hashlib.sha256).digest()
    # Generate the signature
    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
    # Generate the authorization header
    authorization_header = f'AWS4-HMAC-SHA256 Credential={aws_access_key}/{datestamp}/{aws_region}/{aws_service}/aws4_request, ' \
                           f'SignedHeaders=content-type;host;x-amz-date, Signature={signature}'
    # Make the API request
    headers = {
        'Content-Type': 'application/json',
        'Host': endpoint.replace('https://', ''),
        'x-amz-date': timestamp,
        'Authorization': authorization_header
    }
    print("Bodyyyyyy",payload['body'])
    response = requests.post(endpoint + path, headers=headers, data=payload['body'])
    print(response)
    # Process the response
     # Return the image data as JSON response
    return jsonify(response.json())




# Request information
anthropicendpoint = 'https://bedrock.us-east-1.amazonaws.com'
anthropicpath = '/model/anthropic.claude-instant-v1/invoke'
@app.route('/api/call-python3', methods=['POST'])
def call_python3():
    # API payload
    payload = request.json
    
    print("Invking the api----------------------", payload)
    # Generate a timestamp in ISO 8601 format
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    # Generate a date string in YYYYMMDD format
    
    datestamp = datetime.datetime.utcnow().strftime('%Y%m%d')
    # Generate a canonical request
    canonical_request = '\n'.join([
        'POST',
        anthropicpath,
        '',
        'content-type:application/json',
        'host:' + anthropicendpoint.replace('https://', ''),
        'x-amz-date:' + timestamp,
        '',
        'content-type;host;x-amz-date',
        hashlib.sha256(payload['body'].encode('utf-8')).hexdigest()
    ])
    # Generate a string to sign
    string_to_sign = '\n'.join([
        'AWS4-HMAC-SHA256',
        timestamp,
        f'{datestamp}/{aws_region}/{aws_service}/aws4_request',
        hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
    ])
    # Generate the signing key
    key = ('AWS4' + aws_secret_key).encode('utf-8')
    k_date = hmac.new(key, datestamp.encode('utf-8'), hashlib.sha256).digest()
    k_region = hmac.new(k_date, aws_region.encode('utf-8'), hashlib.sha256).digest()
    k_service = hmac.new(k_region, aws_service.encode('utf-8'), hashlib.sha256).digest()
    signing_key = hmac.new(k_service, 'aws4_request'.encode('utf-8'), hashlib.sha256).digest()
    # Generate the signature
    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
    # Generate the authorization header
    authorization_header = f'AWS4-HMAC-SHA256 Credential={aws_access_key}/{datestamp}/{aws_region}/{aws_service}/aws4_request, ' \
                           f'SignedHeaders=content-type;host;x-amz-date, Signature={signature}'
    # Make the API request
    headers = {
        'Content-Type': 'application/json',
        'Host': anthropicendpoint.replace('https://', ''),
        'x-amz-date': timestamp,
        'Authorization': authorization_header
    }
    print(payload['body'])
    response = requests.post(anthropicendpoint + anthropicpath, headers=headers, data=payload['body'])
    print(response)
    # Process the response
    responsedata = response.json()
    print(responsedata)
    #print(responsedata['completion'])
    responsedata = response.json()
    print(responsedata['completion'])
    output_text =responsedata['completion']
    return jsonify(output_text)



# Request information
jurrasicendpoint = 'https://bedrock.us-east-1.amazonaws.com'
jurrasicpath = '/model/ai21.j2-grande-instruct/invoke'
@app.route('/api/call-python4', methods=['POST'])
def call_python4():
    # API payload
    payload = request.json
    print("Invoking the API----------------------", payload)
    # Generate a timestamp in ISO 8601 format
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    # Generate a date string in YYYYMMDD format
    datestamp = datetime.datetime.utcnow().strftime('%Y%m%d')
    # Generate a canonical request
    canonical_request = '\n'.join([
        'POST',
        jurrasicpath,
        '',
        'content-type:application/json',
        'host:' + jurrasicendpoint.replace('https://', ''),
        'x-amz-date:' + timestamp,
        '',
        'content-type;host;x-amz-date',
        hashlib.sha256(json.dumps(payload).encode('utf-8')).hexdigest()
    ])
    # Generate a string to sign
    string_to_sign = '\n'.join([
        'AWS4-HMAC-SHA256',
        timestamp,
        f'{datestamp}/{aws_region}/{aws_service}/aws4_request',
        hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
    ])
    # Generate the signing key
    key = ('AWS4' + aws_secret_key).encode('utf-8')
    k_date = hmac.new(key, datestamp.encode('utf-8'), hashlib.sha256).digest()
    k_region = hmac.new(k_date, aws_region.encode('utf-8'), hashlib.sha256).digest()
    k_service = hmac.new(k_region, aws_service.encode('utf-8'), hashlib.sha256).digest()
    signing_key = hmac.new(k_service, 'aws4_request'.encode('utf-8'), hashlib.sha256).digest()
    # Generate the signature
    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
    # Generate the authorization header
    authorization_header = f'AWS4-HMAC-SHA256 Credential={aws_access_key}/{datestamp}/{aws_region}/{aws_service}/aws4_request, ' \
                           f'SignedHeaders=content-type;host;x-amz-date, Signature={signature}'
    # Make the API request
    headers = {
        'Content-Type': 'application/json',
        'Host': jurrasicendpoint.replace('https://', ''),
        'x-amz-date': timestamp,
        'Authorization': authorization_header
    }
    # Use json.dumps to convert the payload to a string
    response = requests.post(jurrasicendpoint + jurrasicpath, headers=headers, data=json.dumps(payload))
    # Process the response
    responsedata = response.json() 
    completion_data = responsedata['completions'][0]['data']
    print("completion_data",completion_data)
    output_text = completion_data['text']
    print("output_text",output_text)
    return jsonify(output_text)


@app.route('/api/extract-s3-data', methods=['POST'])
def extract_s3_data():
    bucket_url = request.json.get('bucketUrl')
    # Extract data from S3 bucket
    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key,
                             region_name=aws_region)
    print("HIeee", s3_client)
    try:
        response = s3_client.get_object(Bucket=bucket_url)
        print("I am printing response",response.json())
        
        data = response['Body'].read().decode('utf-8')
        print("I am printing data",data)
        return jsonify({'data': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/call-rekognition-api', methods=['POST'])
def call_rekognition_api():
    # Get the image file from the request
    image_file = request.files['imageUpload']
    print("I am printing image file",image_file)
    # Read the image file as bytes
    image_bytes = image_file.read()
    # Create a client for Amazon Rekognition
    rekognition_client = boto3.client('rekognition', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key,
                             region_name=aws_region)
    # Call Amazon Rekognition API to detect labels
    response = rekognition_client.detect_labels(
        Image={'Bytes': image_bytes},
        MaxLabels=10
    )
    # Extract and return the labels from the response
    labels = [label['Name'] for label in response['Labels']]
    # Return the labels as the API response
    return {'labels': labels}   

@app.route('/api/conversation/predict', methods=['POST'])
def predict_conversation():
    # Get the input from the request payload
    payload = request.get_json()
    print(payload)
    # Get the input text from the payload
    body = json.loads(payload['body'])
    input_text = body.get('prompt', '')
    # Set up the conversation chain
    module_path = ".."  # Modify this if needed
    sys.path.append(os.path.abspath(module_path))
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    # Get the AWS credentials
    aws_region = 'us-east-1'
    # Set up the Bedrock client
    bedrock = boto3.client(service_name='bedrock', region_name='us-east-1', endpoint_url='https://bedrock.us-east-1.amazonaws.com', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
    # Set up the language model
    cl_llm = Bedrock(model_id="anthropic.claude-v1", client=bedrock, model_kwargs={"max_tokens_to_sample": 1000})
    # Set up the conversation chain and memory
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=cl_llm, verbose=False, memory=memory)
    # Set up the prompt template
    claude_prompt = PromptTemplate.from_template("""The following is a friendly conversation between a human and an AI.
    The AI is talkative and provides lots of specific details from its context. If the AI does not know
    the answer to a question, it truthfully says it does not know.
    Current conversation:
    {history}
    Human: {input}
    Assistant:
    """)
    conversation.prompt = claude_prompt
    # Generate the prediction
    prediction = conversation.predict(input=input_text)
    print("prediction:", prediction)
    # Return the prediction as a JSON response
    return prediction


# bedrock_client = boto3.client(service_name='bedrock', region_name='us-east-1', endpoint_url='https://bedrock.us-east-1.amazonaws.com', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
# # Initialize Bedrock embeddings
# br_embeddings = BedrockEmbeddings(client=bedrock_client)
# print(f"br_embeddings: {br_embeddings}")
# # Load documents from CSV
# loader = CSVLoader("Prepayment-Monitoring-Report_2023Q1.csv")
# documents_aws = loader.load()
# print(f"Number of documents={len(documents_aws)}")
# # Split and chunk documents
# docs = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400, separator=",").split_documents(documents_aws)
# print(f"Number of documents after split and chunking={len(docs)}")
# # Create FAISS vector store from documents
# vectorstore_faiss_aws = FAISS.from_documents(
#     documents=docs,
#     embedding=br_embeddings
# )
#print(f"vectorstore_faiss_aws: number of elements in the index={vectorstore_faiss_aws.index.ntotal}::")
# Initialize Bedrock embeddings
# Set up the Bedrock client and embeddings
# bedrock_client = boto3.client(
#     service_name='bedrock',
#     region_name='us-east-1',
#     endpoint_url='https://bedrock.us-east-1.amazonaws.com',
#     aws_access_key_id=aws_access_key,
#     aws_secret_access_key=aws_secret_key
# )
# br_embeddings = BedrockEmbeddings(client=bedrock_client)
# # Specify the directory path where the PDF files are located
# pdf_directory = "rag_data"
# # Create a PyPDFDirectoryLoader to load the PDF files
# loader = PyPDFDirectoryLoader(pdf_directory)
# # Load the documents from the PDF files
# documents = loader.load()
# # Split and chunk the documents
# chunk_size = 2000
# chunk_overlap = 400
# separator = ","
# docs = []
# for document in documents:
#     text = document.page_content
#     text_length = len(text)
#     start = 0
#     while start < text_length:
#         end = min(start + chunk_size, text_length)
#         chunk = text[start:end]
#         docs.append(chunk)
#         start += chunk_size - chunk_overlap
# print(f"Number of documents after split and chunking: {len(docs)}")
# # Create a FAISS vector store
# vectorstore_faiss = FAISS(
#     embedding_function=br_embeddings,
#     index=None,  # Specify the index if you have a pre-existing index
#     docstore=InMemoryDocstore(),  # Create a MemoryDocStore as the docstore
#     index_to_docstore_id=None,  # Specify the index_to_docstore_id if you have a pre-existing index_to_docstore_id
# )
# # Prepare documents for indexing
# documents = [Document(page_content=text) for text in docs]
# print(f"Number of documents for indexing: {len(documents)}")
# print(f"First document for indexing: {documents[0]}")
# # Add documents to the vector store
# vectorstore_faiss.add_documents(documents)
# # Update the vector store with the embeddings
# vectorstore_faiss.update_embeddings()
# # Example: Get the number of documents in the vector store
# num_documents = vectorstore_faiss.get_document_count()
# print(f"Number of documents in the vector store: {num_documents}")
###############
bedrock_client = boto3.client(service_name='bedrock', region_name='us-east-1', endpoint_url='https://bedrock.us-east-1.amazonaws.com', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
# # Initialize Bedrock embeddings
br_embeddings = BedrockEmbeddings(client=bedrock_client)
print(f"br_embeddings: {br_embeddings}")
loader = PyPDFDirectoryLoader('./rag_data2')

# Load the documents from the PDF files
documents = loader.load()
# Split and chunk the documents
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
    separators = ["\n\n", "\n", " ", "",',']
)
docs = text_splitter.split_documents(documents)

# Split and chunk documents
print(f"Number of documents after split and chunking={len(docs)}")
# Create FAISS vector store from documents
vectorstore_faiss_aws = FAISS.from_documents(
    documents=docs,
    embedding=br_embeddings
)
print(f"vectorstore_faiss_aws: number of elements in the index={vectorstore_faiss_aws.index.ntotal}::")


# We are also providing a different chat history retriever which outputs the history as a Claude chat (ie including the \n\n)
_ROLE_MAP = {"human": "\n\nHuman: ", "ai": "\n\nAssistant: "}
def _get_chat_history(chat_history):
    buffer = ""
    for dialogue_turn in chat_history:
        if isinstance(dialogue_turn, BaseMessage):
            role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
            buffer += f"\n{role_prefix}{dialogue_turn.content}"
        elif isinstance(dialogue_turn, tuple):
            human = "\n\nHuman: " + dialogue_turn[0]
            ai = "\n\nAssistant: " + dialogue_turn[1]
            buffer += "\n" + "\n".join([human, ai])
        else:
            raise ValueError(
                f"Unsupported chat history format: {type(dialogue_turn)}."
                f" Full chat history: {chat_history} "
            )
    return buffer

# # the condense prompt for Claude
# condense_prompt_claude = PromptTemplate.from_template("""{chat_history}

# Answer only with the new question.


# Human: How would you ask the question considering the previous conversation: {question}


# Assistant: Question:""")

# # recreate the Claude LLM with more tokens to sample - this provide longer responses but introduces some latency
# cl_llm = Bedrock(model_id="anthropic.claude-v1", client=bedrock_client, model_kwargs={"max_tokens_to_sample": 500})
# print("I am there")
# memory_chain = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# print("I am here")
# qa = ConversationalRetrievalChain.from_llm(
#     llm=cl_llm, 
#     retriever=vectorstore_faiss_aws.as_retriever(), 
#     #retriever=vectorstore_faiss_aws.as_retriever(search_type='similarity', search_kwargs={"k": 8}),
#     memory=memory_chain,
#     get_chat_history=_get_chat_history,
#     #verbose=True,
#     condense_question_prompt=condense_prompt_claude, 
#     chain_type='stuff', # 'refine',
#     #max_tokens_limit=300
# )

# # the LLMChain prompt to get the answer. the ConversationalRetrievalChange does not expose this parameter in the constructor
# qa.combine_docs_chain.llm_chain.prompt = PromptTemplate.from_template("""
# {context}


# Human: Use at maximum 3 sentences to answer the question inside the <q></q> XML tags. 

# <q>{question}</q>

# Do not use any XML tags in the answer.

# Assistant:""")


# # Create a function to format the chat history for ConversationalRetrievalChain
# _ROLE_MAP = {"human": "\n\nHuman: ", "ai": "\n\nAssistant: "}
# def _get_chat_history(chat_history):
#     buffer = ""
#     for dialogue_turn in chat_history:
#         if isinstance(dialogue_turn, BaseMessage):
#             role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
#             buffer += f"\n{role_prefix}{dialogue_turn.content}"
#         elif isinstance(dialogue_turn, tuple):
#             human = "\n\nHuman: " + dialogue_turn[0]
#             ai = "\n\nAssistant: " + dialogue_turn[1]
#             buffer += "\n" + "\n".join([human, ai])
#         else:
#             raise ValueError(f"Unsupported chat history format: {type(dialogue_turn)}. Full chat history: {chat_history}")
#     return buffer
# Initialize the Conversational Retrieval Chain
# cl_llm = Bedrock(model_id="anthropic.claude-v1", client=bedrock_client, model_kwargs={"max_tokens_to_sample": 500})
# memory_chain = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# qa = ConversationalRetrievalChain.from_llm(
#     llm=cl_llm,
#     retriever=vectorstore_faiss_aws.as_retriever(),
#     memory=memory_chain,
#     get_chat_history=_get_chat_history,
#     condense_question_prompt=CONDENSE_QUESTION_PROMPT,
#     chain_type='stuff',
# )
chathistory1 = []
_template1 = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT1 = PromptTemplate.from_template(_template1)

prompt_template1 = """Use the following pieces of context to answer the question at the end. If you don't know the answer, use your judgement to answer from your knowledge and be precise. Dont fake the answer.

{context}

Question: {question}
Helpful Answer:"""

@app.route('/api/conversation/predict1', methods=['POST'])
def predict_conversation1():
    # Get the input from the request payload
    #print the langchain version
    payload = request.get_json()
    cl_llm = Bedrock(model_id="anthropic.claude-v1", client=bedrock_client, model_kwargs={"max_tokens_to_sample": 500})
    memory_chain = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
    llm=cl_llm,
    retriever=vectorstore_faiss_aws.as_retriever(),
    memory=memory_chain,
    get_chat_history=_get_chat_history,
    verbose = True,
    condense_question_prompt=CONDENSE_QUESTION_PROMPT1,
    chain_type='stuff',
)

    print(payload)
    # Get the input text from the payload
    body = json.loads(payload['body'])
    question = body.get('prompt', '')
    print("question:",question)
    #question = payload['question']

    trychat = chathistory1
    chat_history = trychat 
    print("chat_history:",chat_history)
    #
    #
    # Append the new question to the chat history
    trychat.append((question, ''))
    # Generate the prediction from the Conversational Retrieval Chain
    print(CONDENSE_QUESTION_PROMPT1.template)
    prediction = qa.run(question=question)
    print("prediction:",prediction)
    
    # Return the prediction as a JSON response
    return jsonify(prediction)



ANTHROPIC_API_KEY = ''
@app.route('/api/conversation/claude100K', methods=['POST'])
def claude100K():
    payload = request.json
    body = json.loads(payload['body'])
    input_text = body.get('prompt', '')
    # api_key = os.environ.get('')
    # if api_key is None:
    #     return jsonify({'error': 'API key not found'}), 500
    client = anthropic.Client(ANTHROPIC_API_KEY)
    response = client.completion(
        prompt=f"{anthropic.HUMAN_PROMPT}{input_text}{anthropic.AI_PROMPT}",
        model="claude-1",
        max_tokens_to_sample=100,
    )
    responsedata = response.json()
    output_text = responsedata['completion']
    return jsonify(output_text)



if __name__ == '__main__':
    app.run(debug=True)