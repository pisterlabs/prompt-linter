import os
from langchain import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.memory import ConversationBufferMemory
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import pinecone
import boto3


S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_BUCKET_HISTORY_KEY_PREFIX = 'history'
S3_BUCKET_TEMPLATES_KEY_PREFIX = 'templates'

HISTORY_PATH = './data/historyData'
INGESTION_PATH = '../data/ingestData'

load_dotenv()
app = Flask(__name__)

chain_instances = {}
conversational_memory_instances = {}

embeddings = OpenAIEmbeddings()
s3_client = boto3.client('s3')
session = boto3.Session()

pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENV')
)
possible_NER_keys = ['influencer', 'user', 'entities', 'talking-style', 'accent', 'slang']


def retrieve_ingestion_template(influencer):
    ner_map = {}
    curr_key = ""
    key_index = 0

    s3_path = f'{S3_BUCKET_TEMPLATES_KEY_PREFIX}/{influencer}'

    if check_path(s3_path, S3_BUCKET_NAME):
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_path)

        # The 'Body' attribute contains the content of the file
        file_content = response['Body'].read().decode('utf-8')

        lines = file_content.splitlines()

        for line in lines:
            if key_index < len(possible_NER_keys) and line.startswith(possible_NER_keys[key_index]):
                curr_key = possible_NER_keys[key_index]
                ner_map[curr_key] = ""
                key_index += 1
            elif len(line) > 0:
                ner_map[curr_key] += line
     
        prompt_template = f"""Use the following pieces of context to answer the question at the end.

        You are acting as {ner_map['influencer']}. Analyze the given conversation in the context. 
        I will talk to you as {ner_map['user']} and you will reply me as {ner_map['influencer']}. 
        Analyze {ner_map['influencer']}'s talking style, tone, and certain slang words that she likes to use and reply to me in  a similar manner. 
        You are to reply to me only once and only as {ner_map['influencer']}. 
        Do not complete the conversation more than once. 
        If there is not enough information, try to infer from the context and reply to me on your own. 
        Try to imitate the talking style provided below sparingly and only when it is appropriate to do so. 

        These are all entities present : 
        {ner_map['entities']}

        This is the talking style of {ner_map['influencer']} : 

        Talking style: {ner_map['talking-style']}

        Accent: {ner_map['accent']}

        Common Slang words: {ner_map['slang']}


        This is the context given:""" + """
        {context}

        """ + f"{ner_map['user']}:" + """
        {question}
        """ + f"""{ner_map['influencer']} :
        """

        return prompt_template


def check_path(file_path, bucket_name):
    result = s3_client.list_objects(Bucket=bucket_name, Prefix=file_path)
    exists = False
    if 'Contents' in result:
        exists = True
    return exists


def retrieve_history(user, influencer, conversational_memory_instance):
    s3_path = f'{S3_BUCKET_HISTORY_KEY_PREFIX}/{influencer}/{user}'

    if check_path(s3_path, S3_BUCKET_NAME):
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_path)

        # The 'Body' attribute contains the content of the file
        file_content = response['Body'].read().decode('utf-8')

        lines = file_content.splitlines()

        for line in lines:
            line_without_prefix = line.replace('Human:', '').replace('AI:', '')

            if line.startswith('Human:'):
                conversational_memory_instance.chat_memory.add_user_message(line_without_prefix)

            elif line.startswith('AI:'):
                conversational_memory_instance.chat_memory.add_ai_message(line_without_prefix)
        print(f'History retrieved for {user} and {influencer}')
        print(conversational_memory_instance.load_memory_variables({}))
        print()
    else:
        print('No history detected.')
        print()


def initialize_chat(user_name, influencer,temperature):
    global chain_instances
    global conversational_memory_instances

    # s3_client.upload_file('./data/fabian_chat.txt', 'digital-immortality-chat-history', 'chat-history/history')
    # Create an index using the influencer name
    index_name = 'test-index'
    if index_name not in pinecone.list_indexes():
        print("Index does not exist, creating a new index called " + index_name)
        pinecone.create_index(
            name=index_name,
            metric="cosine",
            dimension=1536)

    vectorstore = Pinecone.from_existing_index(index_name, embeddings, namespace=influencer)

    # conversational_memory_instance = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt_template = retrieve_ingestion_template(influencer)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": prompt}

    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=temperature,
    )

    current_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(search_kwargs={'k': 3}),
        chain_type="stuff",
        combine_docs_chain_kwargs=chain_type_kwargs,
        verbose=True,)

    chain_instances[(user_name, influencer)] = current_chain
    print(f'Chain for {user_name} and {influencer} formed')
    print('Here are all the current chains so far :')
    print(list(chain_instances.keys()))

def stop_chat_instance(user_name, influencer) :
    chain_instances.pop((user_name,influencer))


@app.route('/startchat', methods=['POST'])
def start_chat():
    user_name = request.args.get('user')
    influencer = request.args.get('influencer')
    temperature = request.args.get('temperature')

    initialize_chat(user_name, influencer,temperature)

    return f'Chat bot initialized for {user_name} and {influencer}', 200

@app.route('/stopchat', methods=['POST'])
def stop_chat():
    user_name = request.args.get('user')
    influencer = request.args.get('influencer')
    stop_chat_instance(user_name, influencer)

    return f'Chat bot stopped for {user_name} and {influencer}', 200


@app.route('/chat', methods=['POST'])
def chat_bot():
    data = request.get_json()
    if not data:
        return jsonify({"Error": "No data provided"}), 400

    user_name = request.args.get('user')
    influencer = request.args.get('influencer')

    messages = data.get("message")
    result = chain_instances[(user_name, influencer)]({"question": messages,  "chat_history" :""})

    return jsonify({"Status": "success", "Message": result["answer"]})


# This API Endpoint will take in a form file, save it to a local directory, and upload to Pinecone
@app.route('/ingest', methods=['POST'])
def ingest_data():
    file = request.files['file']
    index_name = request.args.get('index')
    influencer = request.args.get('influencer')
    file_path = f'{INGESTION_PATH}/{influencer}'

    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print(f'Folder {file_path} created.')

    filename = secure_filename(file.filename)
    filename = os.path.join(file_path, filename)
    file.save(filename)

    documents = []
    loader = TextLoader(file_path=filename, encoding='cp850')
    documents.extend(loader.load())

    # split the documents, create embeddings for them, 
    # and put them in a vectorstore  to do semantic search over them.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    documents = text_splitter.split_documents(documents)
    Pinecone.from_documents(documents, embeddings, index_name=index_name, namespace=influencer)
    os.remove(filename)

    return 'File successfully uploaded.', 200


# Takes in an ingestion template and upload it into local file ./data/ingestData/{fileName}
# Upload the local data onto S3 bucket/influencer
@app.route('/ingest/template', methods=['POST'])
def ingest_template():
    file = request.files['file']
    influencer = request.args.get('influencer')
    file_path = f'{INGESTION_PATH}/{influencer}'

    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print(f'Folder {file_path} created.')

    filename = secure_filename(file.filename)
    filename = os.path.join(file_path, filename)
    file.save(filename)

    s3_client.upload_file(f'{filename}', S3_BUCKET_NAME, f'{S3_BUCKET_TEMPLATES_KEY_PREFIX}/{influencer}')
    os.remove(filename)
    return 'Success', 200


# Write the chat history into a local file ./data/historyData/history.txt and then upload into
# S3 bucket chat-history/{influencer}/{user}
@app.route('/save', methods=['POST'])
def save_history():
    user_name = request.args.get('user')
    influencer = request.args.get('influencer')

    file_path = f'{HISTORY_PATH}/{influencer}/{user_name}'

    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print(f'Folder {file_path} created.')

    # save to local history then upload to aws
    with open(f'{file_path}/history.txt', "w") as file:
        chat_history = conversational_memory_instances[(user_name, influencer)].load_memory_variables({})[
            'chat_history']
        for message in chat_history:
            if hasattr(message, 'content'):
                if 'Human' in str(message.__class__):
                    file.write(str('Human:' + message.content))
                elif 'AI' in str(message.__class__):
                    file.write(str('AI:' + message.content))

    s3_client.upload_file(f'{file_path}/history.txt', S3_BUCKET_NAME,
                          f'{S3_BUCKET_HISTORY_KEY_PREFIX}/{influencer}/{user_name}')
    os.remove(f'{file_path}/history.txt')
    return 'Success', 200


if __name__ == '__main__':
    app.wsgi_app = ProxyFix(
        app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
    )
