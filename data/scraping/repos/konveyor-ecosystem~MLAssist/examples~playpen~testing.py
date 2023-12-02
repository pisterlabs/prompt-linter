 
import os
import dotenv
import time
import subprocess
import sqlite3

import langchain as lc

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor


dotenv.load_dotenv()

def load_github_repo(repo_owner, repo_name, branch):
  repo_path = f"./{repo_owner}/{repo_name}/{branch}"

  if os.path.exists(repo_path):
    loader = GitLoader(
      repo_path=repo_path,
      branch=branch
    )
  else:
    loader = GitLoader(
      clone_url=f"https://github.com/{repo_owner}/{repo_name}",
      repo_path=repo_path,
      branch=branch
    )

  return loader.load()

def chunk_sources(sources):
  splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=16)
  output = []

  for source in sources:
    for chunk in splitter.split_text(source.page_content):
      output.append(Document(page_content=chunk, metadata=source.metadata))

  return output


def connect_to_sqlite(sqlite_db):
  conn = sqlite3.connect(sqlite_db)
  cursor = conn.cursor()

  cursor.execute("""
    CREATE TABLE IF NOT EXISTS llm_summaries (
      id TEXT PRIMARY KEY,
      summary TEXT
    )
  """)

  cursor.execute("""
    CREATE TABLE IF NOT EXISTS file_chunks (
      id TEXT,
      chunk_num INTEGER,
      chunk_content TEXT,
      PRIMARY KEY (id, chunk_num)
    )
  """)

  cursor.execute("""
    CREATE TABLE IF NOT EXISTS number_of_chunks (
      id TEXT PRIMARY KEY,
      chunk_num INTEGER
    )
  """)

  conn.commit()

  return conn

llm = ChatOpenAI(temperature=0)
embeddings = OpenAIEmbeddings()
repo_owner = "konveyor"
repo_name = "spring-framework-petclinic"
repo_branch = "legacy"
# repo_owner = "fabiodomingues"
# repo_name = "javaee-legacy-app-example"
# repo_branch = "master"

repo_path = f"./{repo_owner}/{repo_name}/{repo_branch}"
chroma_directory = f"./{repo_owner}_{repo_name}_{repo_branch}_chroma"
# sqlite_db = f"./{repo_owner}_{repo_name}_{repo_branch}_sqlite.db"
sqlite_db = ":memory:"

sources = load_github_repo(repo_owner, repo_name, repo_branch)
chunks = chunk_sources(sources)
conn = connect_to_sqlite(sqlite_db)

if os.path.exists(chroma_directory):
  search_index = Chroma(
    embedding_function=embeddings, 
    persist_directory=chroma_directory
  )
else:
  search_index = Chroma(
    embedding_function=embeddings, 
    persist_directory=chroma_directory
  )
  for i, chunk in enumerate(chunks):
    print(f"Adding chunk {i+1} of {len(chunks)}")
    search_index.add_documents([chunk])
    time.sleep(0.05)
  search_index.persist()

prompt_template = """Use context the below to answer the question regarding migrating the application to Kubernetes:
Context: {context}
Question: {question}
Answer:"""


PROMPT = PromptTemplate(
  template=prompt_template, 
  input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
  llm=llm,
  chain_type="stuff", 
  retriever=search_index.as_retriever(), 
  return_source_documents=True
)

@lc.agents.tool
def get_project_tree(show_hidden_folders=False, show_hidden_files=False):
  """Returns all directories and subdirectories of the project."""
  tree = ''
  indentation = '\t'
  if not os.path.exists(repo_path):
    return f"Path '{repo_path}' does not exist."

  def explore_directory(current_path, depth):
    nonlocal tree
    files = []
    dirs = []
    with os.scandir(current_path) as entries:
      for entry in entries:
        name = entry.name
        is_dir = entry.is_dir()
        hidden = name.startswith('.')

        if is_dir:
          if hidden and not show_hidden_folders:
            continue
          dirs.append(name)
        else:
          if hidden and not show_hidden_files:
            continue
          files.append(name)
    dirs.sort()
    files.sort()

    for d in dirs:
      tree += f"{indentation * depth}{d}\n"
      explore_directory(os.path.join(current_path, d), depth + 1)
    for f in files:
      tree += f"{indentation * depth}{f}\n"

  tree += repo_path + '\n'
  explore_directory(repo_path, 1)
  return tree

# @lc.agents.tool
# def get_repo_context(query):
#   """Get useful context from the Git repostiory. Information is stored in a vector database. Input should be a fully formed question."""
#   result = qa(query)
#   return result['result']

@lc.agents.tool
def get_summary(file_path):  
  """Gets a LLM-generated short summary of the given file. Note: this is NOT a substitute for looking at the actual contents of the file."""
  if not isinstance(file_path, str) or not file_path.startswith(repo_path):
    return f"Invalid file path. (Note: the root of the current project is at `{repo_path}`. Try listing the project tree to see all available files.)"
  
  cursor = conn.cursor()

  cursor.execute("""
    SELECT summary FROM llm_summaries WHERE id = ?
  """, (file_path,))
  result = cursor.fetchone() 

  if result:
    print(f"Returning cached summary for {file_path}.")
    print(f"{result[0]}")
    return result[0] 
  
  print(f"Generating new summary for {file_path}.")
  llm = ChatOpenAI(temperature=0)
  prompt = f"""
    You are a hyperintelligent software engineer.
    Summarize the following code. Be extremely consise, but do not leave out critical information.
    Someone should be able to recreate the file just from your description alone.
    Always state what frameworks are being used and which version numbers they are.
    {read_file_to_string(file_path)}
  """

  summary = llm.predict(prompt)

  cursor.execute("""
    INSERT INTO llm_summaries (id, summary) VALUES (?, ?)
  """, (file_path, summary))

  conn.commit()

  return summary
  # chain = load_summarize_chain(llm, chain_type="map_reduce")
  # return chain.run([Document(page_content=read_file_to_string(file_path))])

@lc.agents.tool
def read_file_to_string(file_path, chunk_index=1):
  """Takes a file's location as input and prints out the contents of the file. If the text is longer than a single chunk (1024 characters), you must pass in the chunk index (1-indexed)."""
  CHUNK_SIZE = 1024

  try:
    with open(file_path, 'r') as file:
      cursor = conn.cursor()

      cursor.execute("""
        SELECT chunk_num FROM number_of_chunks WHERE id = ?
      """, (file_path,))
      result = cursor.fetchone()

      if result:
        # We have read this file before
        number_of_chunks = result[0]
        print(f"READ FILE BEFORE. NUMBER OF CHUNKS: {number_of_chunks}")
        if chunk_index <= 0 or chunk_index > number_of_chunks:
          return f"chunk_index is outside of the range [1, {number_of_chunks}]"
        
        cursor.execute("""
          SELECT chunk_content FROM file_chunks WHERE id = ? AND chunk_num = ?
        """, (file_path, chunk_index))
        result = cursor.fetchone()

        if result:
          return f"--- Chunk {chunk_index} of {number_of_chunks} for {file_path} ---\n{result[0]}"
        
        return f"--- Error retrieving chunk {chunk_index} of {number_of_chunks} for {file_path} ---"
      
      # We have not read this file before.
      print("NOT READ FILE BEFORE.")
      file_contents = file.read()


      chunks = [file_contents[i:i + CHUNK_SIZE] for i in range(0, len(file_contents), CHUNK_SIZE)]

      cursor.execute("""
        INSERT INTO number_of_chunks (id, chunk_num) VALUES (?, ?)
      """, (file_path, len(chunks)))

      for i, chunk in enumerate(chunks):
        cursor.execute("""
          INSERT INTO file_chunks (id, chunk_num, chunk_content) VALUES (?, ?, ?)
        """, (file_path, i+1, chunk))
      
      return f"--- Chunk {chunk_index} of {len(chunks)} for {file_path} ---\n{chunks[chunk_index-1]}"
  except FileNotFoundError:
    return f"File '{file_path}' not found."
  except IOError:
    return f"Error reading file '{file_path}'."
  
# print(read_file_to_string("./konveyor/spring-framework-petclinic/legacy/readme.md"))
# print(read_file_to_string("./konveyor/spring-framework-petclinic/legacy/readme.md", 2))
# print(read_file_to_string("./konveyor/spring-framework-petclinic/legacy/readme.md", 3))
# exit()

@lc.agents.tool
def execute_git_command(command):
  """Execute a `git` command on the repository. Input should be formatted like `git <command>`."""
  return subprocess.getoutput(f"cd {repo_path} && {command}")

@lc.agents.tool
def regex_search_documentation(query):
  """Performs a regex on the documentation's index for the specified term. Search for method or class names."""

  print(f"SEARCHING DOCUMENTATION FOR {query}")

  lines = []
  while True:
    line = input("> ")
    if not line:
      if lines and lines[-1] == '':
        break
    lines.append(line)
  
  return '\n'.join(lines).strip()

tools = [
  get_project_tree,
  # get_repo_context,
  get_summary,
  read_file_to_string,
  execute_git_command,
  # regex_search_documentation
]

# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

system_message_content = """
You are very intelligent software engineer that helps migrate applications to run on different platforms (e.g. from VMs to OpenShift, bare metal to EKS, etc...). 
You have access to the git repository of an application. 
Your job is to help assist in migrating this application by detecting outdated API usage, any potential problems, etc...
""".strip()

system_message = SystemMessage(content=system_message_content)

MEMORY_KEY = "chat_history"
prompt = OpenAIFunctionsAgent.create_prompt(
  system_message=system_message,
  # extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)]
)

# memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

inp = """
- Trying to migrate to EKS EKS
- Session id is changing with each page load
- Hazelcast not configured properly to work with Kubernetes?
- Cookie management is ok

Please analyze the repository and provide a specific source code modification or configuration change to try. You may need to examine file contents and search documentation.
""".strip()

print(inp)

agent_executor.run(inp)

while True:
  agent_executor.run(input("> "))

# # chain = LLMChain(llm=llm, prompt=PROMPT)

# def answer_question(question):
#   # docs = search_index.similarity_search(question, k=3)
#   # inputs = [{"context": doc.page_content, "question": question} for doc in docs]
#   # print(chain.apply(inputs))



#   res = qa("---------------------\nYou are a hyperintelligent software engineer. Using the documentation provided, assist with the following problem:\nQuestion: " + question + "\nResponse:")
#   answer, docs = res['result'], res['source_documents']
#   res = answer + "\n\n\n" + "Sources:\n"
  
#   sources = set()  # To store unique sources
  
#   # Collect unique sources
#   for document in docs:
#     if "source" in document.metadata:
#       sources.add(document.metadata["source"])
  
#   # Print the relevant sources used for the answer
#   for source in sources:
#     if source.startswith("http"):
#       res += "- " + source + "\n"
#     else:
#       res += "- source code: " + source + "\n"

#   print(res)

# while True:
#   answer_question(input("> "))