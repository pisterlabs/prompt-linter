from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter

import re

import os
import openai
import json
from fileservices import DigitalOceanSpaces
import mysql.connector
from dotenv import load_dotenv
import logging

# Add this near the start of your code
logging.basicConfig(level=logging.ERROR)


load_dotenv()
# import mysql connector


import time

# Reset the MessageHistory
message_history = []





ACCESS_ID = os.environ.get('ACCESS_ID')
SECRET_KEY = os.environ.get('SECRET_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

MYSQL_HOST = os.environ.get('MYSQL_HOST')
MYSQL_PORT = os.environ.get('MYSQL_PORT')
MYSQL_USER = os.environ.get('MYSQL_USER')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD')
MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE')



# Set up the OpenAI API
openai.api_key = OPENAI_API_KEY



# Local API key for testing

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


#pdf = "pdf/Full Dataset.pdf"
dosfile = DigitalOceanSpaces('exon-hosting', 'nyc3', 'https://nyc3.digitaloceanspaces.com', os.environ.get('ACCESS_ID'), os.environ.get('SECRET_KEY'))




def preprocess_and_embed_texts(dataset):
    # Get all files in the PDF folder
    # Download the file
    dosfile.download_file("temp.pdf", dataset)
    loader = UnstructuredPDFLoader("temp.pdf")
    data = loader.load()
    print (f'You have {len(data)} document(s) in your data')
    print (f'There are {len(data[0].page_content)} characters in your document')

    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    print (f'Now you have {len(texts)} documents')

    #print(texts)
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV # next to api key in console
    )
    index_name = "exon-hostings"
    namespace = "text"
    # Time how long it takes to index the documents
    totalStart = time.time()
    start = time.time()
    index = pinecone.Index(index_name)
    index.delete(deleteAll='true', namespace=namespace)
    Pinecone.from_texts(
      [t.page_content for t in texts], embeddings,
      index_name=index_name, namespace=namespace)
    
    # This creates the index in pinecone so that it can be easily called later when a user asks a query.
    
    
    return "Complete"
    
def cx_message_history(clientid, message, addorget):
    with mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        port=MYSQL_PORT,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
        
        
    ) as cnx:
        with cnx.cursor() as cursor:
            cursor.execute("SELECT * FROM cx_messaging WHERE cx_id = %s", (clientid,))
            result = cursor.fetchall()  # Fetch the results first
            print("Result Count: ", len(result))  # Check the number of rows in the result
            if len(result) == 1:
                #print(result)
                if addorget == "add":
                    message_json = json.dumps(message)  # convert list to JSON string

                    cursor.execute(
                        "UPDATE cx_messaging SET cx_messagehistory = %s WHERE cx_id = %s",
                        (message_json, clientid,)
                    )
                    cnx.commit()
                    return True
                elif addorget == "get":
                    return json.loads(result[0][3]) if result[0][3] else None  # convert JSON string to Python list
            else:
                return False

def run_query(query):
    cnx = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        port=MYSQL_PORT,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        ssl_disabled=MYSQL_SSL
    )
    cursor = cnx.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    print("QUERY| Cursor Row Count: ", cursor.rowcount)
    cnx.commit()
    cursor.close()
    cnx.close()
    print(result)

def add_cx_to_db(clientid, phone, name):
    try:
        logging.debug("Trying to connect to the MySQL server...")
        cnx = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            port=MYSQL_PORT,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
            
            
        )
        logging.debug("Connected to the MySQL server.")
    except Exception as e:
        logging.error("Failed to connect to the MySQL server: %s", e)
        return False

    try:
        logging.debug("Trying to get a cursor from the connection...")
        cursor = cnx.cursor()
        logging.debug("Got a cursor.")
    except Exception as e:
        logging.error("Failed to get a cursor from the connection: %s", e)
        return False

    try:
        logging.debug("Executing SELECT statement...")
        cursor.execute("SELECT * FROM cx_messaging WHERE cx_id = %s", (clientid,))
        logging.debug("Executed SELECT statement.")
        if cursor.fetchone() is not None:
            logging.debug("Client ID already exists")
            return False  # client id already exists
        logging.debug("Client ID does not exist")
        logging.debug("SQL: INSERT INTO cx_messaging (cx_phone, cx_name, cx_id, cx_messagehistory) VALUES (%s, %s, %s, '')", (phone, name, clientid))
        logging.debug("Executing INSERT statement...")
        cursor.execute(
            "INSERT INTO cx_messaging (cx_phone, cx_name, cx_id, cx_messagehistory) VALUES (%s, %s, %s, '')",
            (phone, name, clientid)
        )
        logging.debug("Executed INSERT statement.")
        cnx.commit()
        logging.debug("Committed the transaction.")
    except Exception as e:
        logging.error("Failed to execute SQL statement: %s", e)
        return False

    finally:
        # Close your cursor and connection whether it was successful or not
        logging.debug("Closing cursor and connection...")
        cursor.close()
        cnx.close()
        logging.debug("Closed cursor and connection.")

    return True
import time

def gpt4query(prompt):
    
    completion = openai.ChatCompletion.create(model="gpt-4",max_tokens=256,messages=prompt) # Send the prompt to GPT-4 and get a response
    return completion.choices[0].message["content"]

def gpt3query(clientid, query, max_tokens=4096):
    start_time = time.time()
    
    message_history = cx_message_history(clientid, "xxx", "get")
    if message_history is None:
        message_history = []

    # Add the new user message
    message_history.append({"role": "user", "content": query})
    end_time = time.time()
    print(f"Time taken for appending user message: {end_time - start_time} seconds")

    start_time = time.time()
    # Calculate tokens in message history
    total_tokens = sum(len(message["content"]) for message in message_history)

    # If total tokens exceed max tokens, remove oldest messages
    while total_tokens > max_tokens:
        removed_message = message_history.pop(0)  # remove the oldest message
        total_tokens -= len(removed_message["content"])  # update the total tokens
    end_time = time.time()
    print(f"Time taken for calculating tokens and removing old messages: {end_time - start_time} seconds")

    start_time = time.time()
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",max_tokens=256,messages=message_history) # Send the prompt to GPT-3 and get a response
    end_time = time.time()
    print(f"Time taken for GPT-3.5-turbo to respond: {end_time - start_time} seconds")

    start_time = time.time()
    message_history.append({"role": "assistant", "content": completion.choices[0].message["content"]}) # Add the response to the message history

    # Update DB with new message history
    cx_message_history(clientid, message_history, "add")
    end_time = time.time()
    print(f"Time taken for appending assistant message and updating DB: {end_time - start_time} seconds")
    
    return completion.choices[0].message["content"]



def ask(query, phone):
    # return the numbers from the phone
    cx_id = phone.replace("whatsapp:+", "")
    cx_id = cx_id[-5:]
    add_cx_to_db(cx_id, "NA", phone)


    index_name = "exon-hostings"
    namespace = "text"
    start = time.time()
    totalStart = time.time()
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV # next to api key in console
    )
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    # Find the existing index in pinecone so that it can be used for similarity search.

    end = time.time()
    print(f"Indexing took {end - start} seconds")


    #query = "What is webroot?"
    start = time.time()
    docs = docsearch.similarity_search(query,
      namespace=namespace)
    end = time.time()
    print(f"Searching took {end - start} seconds")

    start = time.time()
    print(f"Found {len(docs)} documents")
    #print(docs)
    #print(docs[0].page_content)
    fulltext = ""
    for doc in docs:
        fulltext += doc.page_content
    fulltext = fulltext.replace("\n", " ")
    
    with open("fulltext.txt", "w") as f:
        f.write(fulltext)
    
    end = time.time()
    print(f"Fulltext took {end - start} seconds")
    start = time.time()
    
    assemblePrompt = "You are a Hajj bot, you will help people understand the events and meaning of Hajj, you will answer the question: {} by using the following '{}'. Answer the question in 3 sentances or less.".format(query, fulltext)
    # Format prompt so that there are not non alpha and numeric characters
    
    assemblePrompt = re.sub(r'[^a-zA-Z0-9 ]', '', assemblePrompt)
    print(assemblePrompt)
    answer = gpt3query(cx_id, assemblePrompt)
    #answer = chain.run(input_documents=docs, question=query)
    end = time.time()
    totalEnd = time.time()

    print(f"Answering took {end - start} seconds")
    print(f"Total time: {totalEnd - totalStart} seconds")
    # Format the answer so that it's easier to read and there isnt any extra newlines or spaces before the answer
    answer = answer.replace("\n", "")
    
    return answer


