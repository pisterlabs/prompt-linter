import weaviate
import langchain
import apscheduler
import tempfile
import gradio as gr
from langchain.embeddings import CohereEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import Weaviate
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os
import urllib.request
import ssl
import mimetypes
from dotenv import load_dotenv
import cohere
from apscheduler.schedulers.background import BackgroundScheduler
import time

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI')
cohere_api_key = os.getenv('COHERE')
weaviate_api_key = os.getenv('WEAVIATE')
weaviate_url = os.getenv('WEAVIATE_URL')
weaviate_username = os.getenv('WEAVIATE_USERNAME')
weaviate_password = os.getenv('WEAVIATE_PASSWORD')


# Function to refresh authentication
def refresh_authentication():
    global my_credentials, client
    my_credentials = weaviate.auth.AuthClientPassword(username=weaviate_username, password=weaviate_password)
    client = weaviate.Client(weaviate_url, auth_client_secret=my_credentials)

# Initialize the scheduler for authentication refresh
scheduler = BackgroundScheduler()
scheduler.add_job(refresh_authentication, 'interval', minutes=30)
scheduler.start()

# Initial authentication
refresh_authentication()

Article = {
  "class": "Article",
  "description": "A class representing articles in the application",
  "properties": [
    {
      "name": "title",
      "description": "The title of the article",
      "dataType": ["text"]
    },
    {
      "name": "content",
      "description": "The content of the article",
      "dataType": ["text"]
    },
    {
      "name": "author",
      "description": "The author of the article",
      "dataType": ["text"]
    },
    {
      "name": "publishDate",
      "description": "The date the article was published",
      "dataType": ["date"]
    }
  ],
#  "vectorIndexType": "hnsw",
#  "vectorizer": "text2vec-contextionary"
}

# Function to check if a class exists in the schema
def class_exists(class_name):
    try:
        existing_schema = client.schema.get()
        existing_classes = [cls["class"] for cls in existing_schema["classes"]]
        return class_name in existing_classes
    except Exception as e:
        print(f"Error checking if class exists: {e}")
        return False

# Check if 'Article' class already exists
if not class_exists("Article"):
    # Create the schema if 'Article' class does not exist
    try:
        client.schema.create(schema)
    except Exception as e:
        print(f"Error creating schema: {e}")
else:
    print("Class 'Article' already exists in the schema.")

# Initialize the schema
schema = {
    "classes": [Article]
}

# Check if 'Article' class already exists
if not class_exists("Article"):
    # Create the schema if 'Article' class does not exist
    try:
        client.schema.create(schema)
    except Exception as e:
        print(f"Error creating schema: {e}")
else:
    # Retrieve the existing schema if 'Article' class exists
    try:
        existing_schema = client.schema.get()
        print("Existing schema retrieved:", existing_schema)
    except Exception as e:
        print(f"Error retrieving existing schema: {e}")
        

# Initialize vectorstore
vectorstore = Weaviate(client, index_name="HereChat", text_key="text")
vectorstore._query_attrs = ["text", "title", "url", "views", "lang", "_additional {distance}"]
vectorstore.embedding = CohereEmbeddings(model="embed-multilingual-v2.0", cohere_api_key=cohere_api_key)

# Initialize Cohere client
co = cohere.Client(api_key=cohere_api_key)

def embed_pdf(file, filename, collection_name, file_type):
    # Check the file type and handle accordingly
    if file_type == "URL":
        # Download the file from the URL
        try:
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(file, context=context) as response, open(filename, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            file_path = filename
        except Exception as e:
            return {"error": f"Error downloading file from URL: {e}"}
    elif file_type == "Binary":
        # Handle binary file
        if isinstance(file, str):
            # Convert string to bytes if necessary
            file = file.encode()
        file_content = file
        file_path = os.path.join('./', filename)
        with open(file_path, 'wb') as f:
            f.write(file_content)
    else:
        return {"error": "Invalid file type"}


    # Checking filetype for document parsing
    mime_type = mimetypes.guess_type(file_path)[0]
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()

    # Generate embeddings and store documents in Weaviate
    embeddings = CohereEmbeddings(model="embed-multilingual-v2.0", cohere_api_key=cohere_api_key)
    for doc in docs:
        embedding = embeddings.embed([doc['text']])
        weaviate_document = {
            "text": doc['text'],
            "embedding": embedding
        }
        client.data_object.create(data_object=weaviate_document, class_name=collection_name)

    # Clean up if a temporary file was created
    if isinstance(file, bytes):
        os.remove(file_path)
    return {"message": f"Documents embedded in Weaviate collection '{collection_name}'"}

def retrieve_info(query):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    
    # Retrieve initial results
    initial_results = qa({"query": query})

    # Assuming initial_results are in the desired format, extract the top documents
    top_docs = initial_results[:25]  # Adjust this if your result format is different

    # Rerank the top results
    reranked_results = co.rerank(query=query, documents=top_docs, top_n=3, model='rerank-english-v2.0')

    # Format the reranked results according to the Article schema
    formatted_results = []
    for idx, r in enumerate(reranked_results):
        formatted_result = {
            "Document Rank": idx + 1,
            "Title": r.document['title'],  
            "Content": r.document['content'],  
            "Author": r.document['author'],  
            "Publish Date": r.document['publishDate'],  
            "Relevance Score": f"{r.relevance_score:.2f}"
        }
        formatted_results.append(formatted_result)
        
    return {"results": formatted_results}
        # Format the reranked results and append to user prompt
    user_prompt = f"User: {query}\n"
    for idx, r in enumerate(reranked_results):
        user_prompt += f"Document {idx + 1}: {r.document['text']}\nRelevance Score: {r.relevance_score:.2f}\n\n"

    # Final API call to OpenAI
    final_response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a redditor. Assess, rephrase, and explain the following. Provide long answers. Use the same words and language you receive."
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=1.63,
        max_tokens=2240,
        top_p=1,
        frequency_penalty=1.73,
        presence_penalty=1.76
    )

    return final_response.choices[0].text

def combined_interface(query, file, collection_name):
    if query:
        article_info = retrieve_info(query)
        return article_info
    elif file is not None and collection_name:
        filename = file[1]  # Extract filename
        file_content = file[0]  # Extract file content

        # Check if file_content is a URL or binary data
        if isinstance(file_content, str) and file_content.startswith("http"):
            file_type = "URL"
            # Handle URL case (if needed)
        else:
            file_type = "Binary"
            # Write binary data to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                temp_file.write(file_content)
                temp_filepath = temp_file.name

            # Pass the file path to embed_pdf
            result = embed_pdf(temp_filepath, collection_name)

            # Clean up the temporary file
            os.remove(temp_filepath)

            return result
    else:
        return "Please enter a query or upload a PDF file and specify a collection name."


iface = gr.Interface(
    fn=combined_interface,
    inputs=[
        gr.Textbox(label="Query"),
        gr.File(label="PDF File"),
        gr.Textbox(label="Collection Name")
    ],
    outputs="text"
)

iface.launch()