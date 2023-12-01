from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader
from flask import Flask, render_template, request, redirect, url_for
from github import Github
from ignore import should_ignore
import openai
import os
import shutil
import subprocess
import pinecone

app = Flask(__name__)

model_id = "gpt-3.5-turbo"
openai.api_key = "<Key>"
os.environ["OPENAI_API_KEY"] = "<Key>"

def langchain_response(question):
    loader = DirectoryLoader(path="./summaries", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    #Print all the document names that are loaded
    print(documents)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    
    pinecone.init(
        api_key="<Key>",  # find at app.pinecone.io
        environment="<Key>",  # next to api key in console
    )

    index_name = "readmeai"
    # The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
    docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)

    llm = OpenAI(temperature=0.1, openai_api_key="<Key>")
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = docsearch.similarity_search(question)
    print(chain.run(input_documents=docs, question=question))
    
    

def ChatGPT_conversation(conversation):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=conversation
    )
    conversation.append({'role': 'AI', 'content': response.choices[0].message.content})
    return conversation

def get_repo_files_recursive(folder):
    repo_files = []

    for entry in os.scandir(folder):
        if entry.name == ".git":
            continue  # Skip the .git folder

        if entry.is_file():
            file_info = {"path": os.path.join(entry.path), "name": entry.name}
            repo_files.append(file_info)
        elif entry.is_dir():
            subdir = os.path.join(entry.path)
            repo_files.extend(get_repo_files_recursive(subdir))

    return repo_files


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        github_url = request.form.get('github_url')
        repo_name = github_url.split("/")[-1].replace(".git", "")
        repo_folder = os.path.join('clone', repo_name)

        # Clone the repository
        subprocess.run(["git", "clone", github_url, repo_folder])

        # Process the cloned files
        if not os.path.exists('summaries'):
            os.mkdir('summaries')

        files = get_repo_files_recursive(repo_folder)
        
        for file in files:
            if not should_ignore(file['name']):
                content = file['name']
                conversation = [
                    {"role": "user", "content": f"Explain the contents of the {file['name']} file in 250 words: {content}"}
                ]
                response = ChatGPT_conversation(conversation)
                with open(f"summaries/{file['name']}.txt", "w") as f:
                    f.write(response[-1]["content"])

                # Print the files and its source after its sent to summarization
                print(f"File: {file['name']} from {file['path']}")

        # Perform Langchain and cleanup
        question = request.form.get('question')
        langchain_response(question)

        # Clean up the cloned repository folder
        # shutil.rmtree(repo_folder)
        # shutil.rmtree('summaries')

        return redirect(url_for('index'))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)