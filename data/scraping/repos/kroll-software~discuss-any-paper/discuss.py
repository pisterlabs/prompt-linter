import os
import pathlib
import glob
import time
import logging
from datetime import datetime
from collections import deque
from typing import Dict, List
import importlib
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains.question_answering import load_qa_chain
import chromadb
from chromadb.utils.embedding_functions import InstructorEmbeddingFunction

# Load default environment variables (.env)
load_dotenv()

# **** Logging output to File

# Engine configuration
LLM_MODEL = "GPT4All"

# Table config
RESULTS_STORE_NAME = os.getenv("RESULTS_STORE_NAME", os.getenv("TABLE_NAME", ""))
assert RESULTS_STORE_NAME, "\033[91m\033[1m" + "RESULTS_STORE_NAME environment variable is missing from .env" + "\033[0m\033[0m"

# Run configuration
INSTANCE_NAME = os.getenv("INSTANCE_NAME", os.getenv("BABY_NAME", "Paper Discussion"))
COOPERATIVE_MODE = "none"
JOIN_EXISTING_OBJECTIVE = False

# Goal configuation
OBJECTIVE = os.getenv("OBJECTIVE", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))

# Model configuration
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))

VERBOSE = (os.getenv("VERBOSE", "false").lower() == "true")

# Extensions support begin

def can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

print("\033[95m\033[1m"+"\n*****CONFIGURATION*****\n"+"\033[0m\033[0m")
print(f"Name  : {INSTANCE_NAME}")
#print(f"Mode  : {'alone' if COOPERATIVE_MODE in ['n', 'none'] else 'local' if COOPERATIVE_MODE in ['l', 'local'] else 'distributed' if COOPERATIVE_MODE in ['d', 'distributed'] else 'undefined'}")
#print(f"LLM   : {LLM_MODEL}")

# Check if we know what we are doing
assert OBJECTIVE, "\033[91m\033[1m" + "OBJECTIVE environment variable is missing from .env" + "\033[0m\033[0m"
assert INITIAL_TASK, "\033[91m\033[1m" + "INITIAL_TASK environment variable is missing from .env" + "\033[0m\033[0m"

MODEL_PATH = os.getenv("MODEL_PATH", "models/gpt4all-lora-quantized-ggml.bin")
    
print(f"LLM : {MODEL_PATH}" + "\n")
assert os.path.exists(MODEL_PATH), "\033[91m\033[1m" + f"Model can't be found." + "\033[0m\033[0m"

#CTX_MAX = 2048
#CTX_MAX = 8192
CTX_MAX = 16384
THREADS_NUM = 4
MAX_TOKENS = 256

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=MODEL_PATH,
    callback_manager=callback_manager,
    n_threads=THREADS_NUM,
    temperature=TEMPERATURE,
    n_ctx=CTX_MAX,  
    #max_tokens=MAX_TOKENS,
    verbose=False,
    #echo=False,
    streaming=False,
)
llm.client.verbose = False

embeddings = HuggingFaceInstructEmbeddings()

# *** Document Loader ***

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 20,
    length_function = len,    
)

def load_docs():
    print("loading documents ..")
    global source_files
    source_files = glob.glob('./docs/*.*')
    docs = []
    for f in source_files:
        loader = None
        ext = pathlib.Path(f).suffix.lower()
        if ext == '.txt':
            loader = UnstructuredFileLoader(f)
        elif ext == '.pdf':
            loader = UnstructuredPDFLoader(f)
        if loader != None:
            pages = loader.load_and_split(text_splitter=text_splitter)
            docs.extend(pages)
    global paper_db
    print("indexing documents ..")
    paper_db = Chroma.from_documents(docs, embeddings)

load_docs()

print("\033[94m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
print(f"{OBJECTIVE}")

if not JOIN_EXISTING_OBJECTIVE: print("\033[93m\033[1m" + "\nInitial task:" + "\033[0m\033[0m" + f" {INITIAL_TASK}")
else: print("\033[93m\033[1m" + f"\nJoining to help the objective" + "\033[0m\033[0m")

# Results storage using local ChromaDB
class DefaultResultsStorage:
    def __init__(self):
        logging.getLogger('chromadb').setLevel(logging.ERROR)
        # Create Chroma collection
        chroma_persist_dir = "chroma"
        chroma_client = chromadb.Client(
            settings=chromadb.config.Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=chroma_persist_dir,
            )
        )

        metric = "cosine"        
        embedding_function = InstructorEmbeddingFunction()
        self.collection = chroma_client.get_or_create_collection(
            name=RESULTS_STORE_NAME,
            metadata={"hnsw:space": metric},
            embedding_function=embedding_function,
        )

    def add(self, task: Dict, result: Dict, result_id: str, vector: List):        
        embeddings = self.collection._embedding_function([vector])        

        if (len(self.collection.get(ids=[result_id], include=[])["ids"]) > 0):  # Check if the result already exists
            self.collection.update(
                ids=result_id,
                embeddings=embeddings,
                documents=vector,
                metadatas={"task": task["task_name"], "result": result},
            )
        else:
            self.collection.add(
                ids=result_id,
                embeddings=embeddings,
                documents=vector,
                metadatas={"task": task["task_name"], "result": result},
            )

    def query(self, query: str, top_results_num: int) -> List[dict]:
        count: int = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=query,
            n_results=min(top_results_num, count),
            include=["metadatas"]
        )        
        tasks = []
        count = len(results["ids"][0])
        for i in range(count):            
            resultidstr = results["ids"][0][i]            
            id = int(resultidstr[7:])
            item = results["metadatas"][0][i]            
            task = {'task_id': id, 'task_name': item["task"]}
            tasks.append(task)            
        return tasks
   

# Initialize results storage
results_storage = DefaultResultsStorage()

# Task storage supporting only a single instance of BabyAGI
class SingleTaskListStorage:
    def __init__(self):
        self.tasks = deque([])
        self.task_id_counter = 0

    def append(self, task: Dict):
        self.tasks.append(task)

    def replace(self, tasks: List[Dict]):
        self.tasks = deque(tasks)

    def popleft(self):
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def next_task_id(self):
        self.task_id_counter += 1
        return self.task_id_counter

    def get_task_names(self):
        return [t["task_name"] for t in self.tasks]


# Initialize tasks storage
tasks_storage = SingleTaskListStorage()

def gpt_call(prompt: str):
    result = llm(prompt[:CTX_MAX])
    return result.strip()
    #return result['choices'][0]['text'][len(prompt):].strip() 

def strip_numbered_list(nl: List[str]) -> List[str]:
    result_list = []
    filter_chars = ['#', '*', '(', ')', '[', ']', '.', ':', ' ']

    for line in nl:
        line = line.strip()
        if len(line) > 0:
            parts = line.split(" ", 1)
            if len(parts) == 2:
                left_part = ''.join(x for x in parts[0] if not x in filter_chars)
                if left_part.isnumeric():
                    result_list.append(parts[1].strip())
                else:
                    result_list.append(line)
            else:
                result_list.append(line)

    # filter result_list
    result_list = [line for line in result_list if len(line) > 3]
    
    # remove duplicates
    result_list = list(set(result_list))
    return result_list

def fix_prompt(prompt: str) -> str:
    lines = prompt.split("\n") if "\n" in prompt else [prompt]    
    return "\n".join([line.strip() for line in lines])


# ************** AGENTS ***************

task_creation_chain = LLMChain.from_string(
    llm, 
    fix_prompt("""
Your objective: {objective}\n
Take into account these previously completed questions but don't repeat them: {task_list}.\n
Excerpts from the paper: {excerpts}\n
The last completed question has the result: {result}.\n
Develop a list of intelligent questions based on the result.\n
Response:""")
)

def task_creation_agent(
    objective: str, result: Dict, task_description: str, task_list: List[str]
):      
    docs = paper_db.similarity_search(result["data"], k=4)
    excerpts = [doc.page_content.replace("\n", " ").replace("  ", " ") for doc in docs]

    response = task_creation_chain.run(objective=objective, task_list=task_list, result=result["data"], excerpts=excerpts)

    pos = response.find("1")
    if (pos > 0):
        response = response[pos - 1:]

    if response == '':
        print("\n*** Empty Response from task_creation_agent***")
        new_tasks_list = result["data"].split("\n") if len(result) > 0 else [response]
    else:
        new_tasks = response.split("\n") if "\n" in response else [response]
        new_tasks_list = strip_numbered_list(new_tasks)
        
    return [{"task_name": task_name} for task_name in (t for t in new_tasks_list if not t == '')]


prioritization_chain = LLMChain.from_string(
    llm, 
    fix_prompt("""
Please prioritize, summarize and consolidate the following questions: {task_names}.\n
Purge incomplete questions.\n
Consider the ultimate objective: {objective}.\n
Return the result in logical order as a numbered list:
""")
)

def prioritization_agent():
    task_names = tasks_storage.get_task_names()
    next_task_id = tasks_storage.next_task_id()    

    response = prioritization_chain.run(task_names=task_names, objective=OBJECTIVE)

    pos = response.find("1")
    if (pos > 0):
        response = response[pos - 1:]

    new_tasks = response.split("\n") if "\n" in response else [response]
    new_tasks = strip_numbered_list(new_tasks)
    new_tasks_list = []
    i = 0
    for task_string in new_tasks:        
        new_tasks_list.append({"task_id": i + next_task_id, "task_name": task_string})
        i += 1
    
    if len(new_tasks_list) > 0:
        tasks_storage.replace(new_tasks_list)


initial_execution_chain = LLMChain.from_string(
    llm, 
    fix_prompt("""
You are an AI who performs one task based on the following objective: {objective}.\n
Your task: {task}\nResponse:""")
)

execution_chain = LLMChain.from_string(
    llm, 
    fix_prompt("""
Your objective: {objective}.\n
Excerpts from the paper: {excerpts}\n
Take into account these previously completed questions but don't repeat them: {context_list}.\n
Your question: {task}\n
Detailed and helpful answer to your question:""")
)

# Execute a task based on the objective and five previous tasks
def execution_agent(objective: str, task: str) -> str:    
    context = context_agent(query=objective, top_results_num=5)

    context_list = [t['task_name'] for t in context if t['task_name'] != INITIAL_TASK]
    #context_list = [t['task_name'] for t in context]

    # remove duplicates
    context_list = list(set(context_list))    

    if VERBOSE and len(context_list) > 0:
        print("\n*******RELEVANT CONTEXT******\n")
        print(context_list)
    
    if task == INITIAL_TASK:
        result = initial_execution_chain.run(objective=objective, task=task)
    else:        
        docs = paper_db.similarity_search(task, k=4)
        excerpts = [doc.page_content.replace("\n", " ").replace("  ", " ") for doc in docs]
        result = execution_chain.run(objective=objective, context_list=context_list, task=task, excerpts=excerpts)    
    
    return result.strip()


# Get the top n completed tasks for the objective
def context_agent(query: str, top_results_num: int):
    """
    Retrieves context for a given query from an index of tasks.

    Args:
        query (str): The query or objective for retrieving context.
        top_results_num (int): The number of top results to retrieve.

    Returns:
        list: A list of tasks as context for the given query, sorted by relevance.

    """
    results = results_storage.query(query=query, top_results_num=top_results_num)
    #print("\n***** RESULTS *****")
    #print(results)
    return results

# Add the initial task if starting new objective
if not JOIN_EXISTING_OBJECTIVE:
    initial_task = {
        "task_id": tasks_storage.next_task_id(),
        "task_name": INITIAL_TASK
    }
    tasks_storage.append(initial_task)

def setup_filelogger():
    global file_logger
    file_logger = logging.getLogger()
    file_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    if not os.path.exists("./outputs"):        
        os.makedirs("./outputs")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    logfile_name = f"./outputs/{timestr}-output.txt"
    file_handler = logging.FileHandler(logfile_name)
    file_handler.setLevel(logging.INFO)
    #file_handler.setFormatter(formatter)
    file_logger.addHandler(file_handler)

def main ():

    setup_filelogger()
    file_logger.info(msg=f"LLM: {os.path.basename(MODEL_PATH)}\n")

    file_logger.info(msg="Documents:")
    for f in source_files:
        file_logger.info(msg=os.path.basename(f))

    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    file_logger.info(msg=f"\nDate: {date_time}")

    file_logger.info(msg="\n*****OBJECTIVE*****\n")
    file_logger.info(msg=OBJECTIVE)    

    question_number = 0

    while not tasks_storage.is_empty():    
        question_number += 1

        # Print the task list
        print("\033[95m\033[1m" + "\n***** QUESTIONS *****\n" + "\033[0m\033[0m")
        for t in tasks_storage.get_task_names():
            print(" • "+t)

        # Step 1: Pull the first incomplete task
        task = tasks_storage.popleft()
        print("\033[92m\033[1m" + "\n***** NEXT QUESTION *****\n" + "\033[0m\033[0m")
        print(task['task_name'])
        file_logger.info(msg=f"\n***** QUESTION #{question_number} *****\n")
        file_logger.info(msg=task['task_name'])

        # Send to execution function to complete the task based on the context
        result = execution_agent(OBJECTIVE, task["task_name"])
        print("\033[93m\033[1m" + "\n***** ANSWER *****\n" + "\033[0m\033[0m")            
        print(result)
        file_logger.info(msg="\n***** ANSWER *****\n")
        file_logger.info(msg=result)

        # Step 2: Enrich result and store in the results storage
        # This is where you should enrich the result if needed
        enriched_result = {
            "data": result
        }  
        # extract the actual result from the dictionary
        # since we don't do enrichment currently
        vector = enriched_result["data"]  

        result_id = f"result_{task['task_id']}"
        results_storage.add(task, result, result_id, vector)

        # Step 3: Create new tasks and reprioritize task list
        # only the main instance in cooperative mode does that
        new_tasks = task_creation_agent(
            OBJECTIVE,
            enriched_result,
            task["task_name"],
            tasks_storage.get_task_names(),
        )

        for new_task in new_tasks:
            if not new_task['task_name'] == '':
                new_task.update({"task_id": tasks_storage.next_task_id()})
                tasks_storage.append(new_task)

        if not JOIN_EXISTING_OBJECTIVE: prioritization_agent()

        # Sleep a bit before checking the task list again
        time.sleep(5)

if __name__ == "__main__":
    main()
