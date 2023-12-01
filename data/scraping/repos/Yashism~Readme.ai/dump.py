# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import TextLoader
# from langchain.document_loaders import DirectoryLoader
# from flask import Flask, render_template, request, redirect, url_for
# from github import Github
# from ignore import should_ignore
# import openai
# import os
# import shutil

# app = Flask(__name__)

# model_id = "gpt-3.5-turbo"
# openai.api_key = "<Key>"
# os.environ["OPENAI_API_KEY"] = "<Key>"

# def clone_repo(repo_url):
#     temp_dir = "/tmp/cloned_repo"
#     if os.path.exists(temp_dir):
#         shutil.rmtree(temp_dir)
#     os.mkdir(temp_dir)
    
#     repo = git.Repo.clone(repo_url, temp_dir)
#     return repo

# def process_llm_response(llm_response):
#     print(llm_response['result'])
#     print('\n\nSources:')
#     for source in llm_response["source_documents"]:
#         print(source.metadata['source'])

# def langchain_response(query):
#     loader = DirectoryLoader(path="./summaries", glob="*.txt", loader_cls=TextLoader)
#     documents = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     texts = text_splitter.split_documents(documents)

#     print("Texts loaded:", texts)  # Print loaded texts

#     persist_directory = 'db'
#     embedding = OpenAIEmbeddings()

#     vectordb = Chroma(
#         persist_directory=persist_directory, 
#         embedding_function=embedding
#     )

#     print("Chroma DB created")

#     retriever = vectordb.as_retriever()
    
#     turbo_llm = ChatOpenAI(
#         temperature=0.5,
#         model_name='gpt-3.5-turbo'
#     )

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=turbo_llm, 
#         chain_type="stuff", 
#         retriever=retriever,
#         return_source_documents=True
#     )

#     llm_response = qa_chain(query)
#     # process_llm_response(llm_response)
#     print(llm_response)

#     return llm_response

# def ChatGPT_conversation(conversation):
#     response = openai.ChatCompletion.create(
#         model=model_id,
#         messages=conversation
#     )
#     conversation.append({'role': 'AI', 'content': response.choices[0].message.content})
#     return conversation

# def get_repo_files(repo):
#     files = []
#     for file in repo.get_contents(""):  # Get root directory contents
#         if file.type == "file":
#             files.append(file)
#     return files

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         github_url = request.form.get('github_url')
#         # Fetch repository data using PyGithub
#         g = Github()
#         repo = g.get_repo(github_url.replace("https://github.com/", ""))
        
#         files = get_repo_files(repo)
        
#         # Create the summaries directory if it doesn't exist
#         if not os.path.exists('summaries'):
#             os.mkdir('summaries')
        
#         # Print summaries for each file
#         for file in files:
#             if not should_ignore(file.name):
#                 content = file.decoded_content.decode("utf-8")
#                 conversation = [
#                     {"role": "user", "content": f"Explain the contents of the {file.name} file in 500 words: {content}"}
#                 ]
#                 response = ChatGPT_conversation(conversation)
#                 #save in a textfile
#                 with open(f"summaries/{file.name}.txt", "w") as f:
#                     f.write(response[-1]["content"])
#         # Redirect to home page
#         langchain_response("What is the code about?")
        
#         #delete the summaries directory
#         shutil.rmtree('summaries')
        
#         return redirect(url_for('index'))
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)
