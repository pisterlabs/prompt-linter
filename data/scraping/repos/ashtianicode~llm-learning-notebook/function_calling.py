#%% 

import llama_index
from llama_index.tools import BaseTool, FunctionTool
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.vector_stores import ChromaVectorStore
from llama_index import StorageContext, VectorStoreIndex
import chromadb
import phoenix as px

#%%

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b * -1  
multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return (a + b ) * -1
add_tool = FunctionTool.from_defaults(fn=add)

from IPython import get_ipython

def execute_code(code: str):
    """Executes the given python code in ipython"""
    ipython = get_ipython()
    ipython.run_code(code)
execute_code_tool = FunctionTool.from_defaults(fn=execute_code)


import os
import shutil
import subprocess

from typing import Any, Optional

def create_directory(directory_name: str) -> None:
    """
    Create a new directory.
    """
    os.makedirs(directory_name, exist_ok=True)

def write_file(file_path: str, content: str) -> None:
    """
    Write content to a file.
    """
    with open(file_path, 'w') as f:
        f.write(content)

def read_file(file_path: str) -> str:
    """
    Read content from a file.
    """
    with open(file_path, 'r') as f:
        return f.read()

def initialize_git(directory_name: str) -> None:
    """
    Initialize a new git repository.
    """
    subprocess.run(["git", "init"], cwd=directory_name)

def git_add_all(directory_name: str) -> None:
    """
    Add all changes to git.
    """
    subprocess.run(["git", "add", "."], cwd=directory_name)

def git_commit(directory_name: str, message: str) -> None:
    """
    Commit changes to git.
    """
    subprocess.run(["git", "commit", "-m", message], cwd=directory_name)

def git_push(directory_name: str, remote: str, branch: str) -> None:
    """
    Push changes to remote repository.
    """
    subprocess.run(["git", "push", remote, branch, "--force"], cwd=directory_name)


def natural_lang_query_github_repo(repo_natural_question_query: str) -> str:
    """
    Ask questions about github repo in natural language about different files. 
    Use this function as a way to read the entire repo and ask specific questions to understand latest state of the repo and the files. 
    As you write new files to git, you can use this function to map out what's the latest state. 
    """

    import os
    from llama_index import download_loader
    from llama_hub.github_repo import GithubRepositoryReader, GithubClient
    download_loader("GithubRepositoryReader")

    github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
    loader = GithubRepositoryReader(
        github_client,
        owner =                  "ashtianicode",
        repo =                   "llm-learning-notebook",
        verbose =                True,
        concurrent_requests =    10,
    )

    docs = loader.load_data(branch="main")

    for doc in docs:
        print(doc.extra_info)

    from llama_index import download_loader, GPTVectorStoreIndex
    index = GPTVectorStoreIndex.from_documents(docs)

    query_engine = index.as_query_engine(top_k=5)
    response = query_engine.query(repo_natural_question_query)
    return response




def natural_lang_query_website_reader(url: str, question:str) -> str:
    from llama_index import VectorStoreIndex, SimpleWebPageReader
    documents = SimpleWebPageReader(html_to_text=True).load_data(
        [url]
    )
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return response





# def read_from_vectordb(collection_name: str, prompt: str):
#     """
#     Read from vectordb.
#     """
#     px.launch_app()
#     llama_index.set_global_handler("arize_phoenix")

#     chroma_client = chromadb.PersistentClient()
#     chroma_collection = chroma_client.get_collection(collection_name)
#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)    
#     index = VectorStoreIndex(storage_context=storage_context)
#     nodes = index.retrieve(prompt, similarity_top_k=3)
#     return nodes

# def write_to_vectordb(collection_name: str, text: str):
#     """
#     Write to vectordb.
#     """
#     px.launch_app()
#     llama_index.set_global_handler("arize_phoenix")

#     chroma_client = chromadb.PersistentClient()
#     chroma_collection = chroma_client.get_or_create_collection(collection_name)
#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)    
#     index = VectorStoreIndex.from_documents([text], storage_context=storage_context, show_progress=True)
#     index.storage_context.persist()




create_directory_tool = FunctionTool.from_defaults(fn=create_directory)
write_file_tool = FunctionTool.from_defaults(fn=write_file)
read_file_tool = FunctionTool.from_defaults(fn=read_file)
initialize_git_tool = FunctionTool.from_defaults(fn=initialize_git)
git_add_all_tool = FunctionTool.from_defaults(fn=git_add_all)
git_commit_tool = FunctionTool.from_defaults(fn=git_commit)
git_push_tool = FunctionTool.from_defaults(fn=git_push)
natural_lang_query_github_repo_tool = FunctionTool.from_defaults(fn=natural_lang_query_github_repo)
natural_lang_query_website_reader_tool = FunctionTool.from_defaults(fn=natural_lang_query_website_reader)


# read_from_vectordb_tool = FunctionTool.from_defaults(fn=read_from_vectordb)
# write_to_vectordb_tool = FunctionTool.from_defaults(fn=write_to_vectordb)

#%%
llm = OpenAI(model="gpt-3.5-turbo")

agent = OpenAIAgent.from_tools([
    multiply_tool,
    add_tool,
    execute_code_tool,
    write_file_tool,
    read_file_tool,
    git_add_all_tool,
    git_commit_tool,
    git_push_tool,
    natural_lang_query_github_repo_tool,
    natural_lang_query_website_reader_tool
    ], llm=llm, verbose=True)

agent.chat("""
    You are studying pandas API. 
    You must take study notes on github using the git tools available to you. 
    Start making a corriculum based on https://pandas.pydata.org/docs/user_guide/10min.html using the webtool to extract all topics of practice. 
    Then create a seperate .py file for each example snippet that you run for practice. 
    Use the execute_code_tool tool for running your code.
    Get the results of your running code and add the result in comment form to the end of your practice file. 
    After each practice, push that file to git. 
    Do your practice one step at a time. 
""")

# %%
