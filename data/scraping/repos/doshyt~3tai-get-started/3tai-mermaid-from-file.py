import openai
import os
import pickle

from config import OPENAI_API_KEY, GITHUB_TOKEN, SYSTEM_PROMPT

from llama_index import Document, GPTVectorStoreIndex, ServiceContext, StorageContext, VectorStoreIndex, download_loader, load_index_from_storage
download_loader("GithubRepositoryReader")
from llama_index.llms import OpenAI
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ContextRetrieverOpenAIAgent
from llama_index import download_loader, set_global_service_context
MarkdownReader = download_loader("MarkdownReader")

from pathlib import Path

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['GITHUB_TOKEN'] = GITHUB_TOKEN

DOCS_INDEX_DIR = './docs_index'
SHORT_SYSTEM_DESC = "" #  replace with your system description one-liner, i.e. Barbario, AI-powered barber and stylist robot"

LLM = OpenAI(temperature=0.5, model="gpt-3.5-turbo-16k")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
openai.api_key = os.getenv("OPENAI_API_KEY")

service_context = ServiceContext.from_defaults(llm=LLM)
set_global_service_context(service_context)

LOADER = MarkdownReader()

def load_index_from_file(loader, index_dir):
    """
    function to generate new index (context) or load existing one from file
    """
    docs = None
    index = None
    
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    
    if os.path.exists(index_dir+'/docs.pkl'):
        with open(index_dir+"/docs.pkl", "rb") as f:
            docs = pickle.load(f)

    if docs is None:
        loader = loader
        docs = loader.load_data(file=Path('./tech_design.md')) # add path to file(s) you want to load - can be code, best practice, tech design.
        
        with open(index_dir+"/docs.pkl", "wb") as f:
                pickle.dump(docs, f)

    if os.path.exists(index_dir) and os.path.isdir(index_dir) and os.path.exists(index_dir+'./docstore.json'):
        print("[INFO] Loading existing index")
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
    else:
        print("[INFO] Building new index")   
        index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
        index.storage_context.persist(persist_dir=index_dir)
    
    return index

def prepare_agent():
    """
    function to prepare agent that will interact with LLM
    """
    system_docs_index = load_index_from_file(LOADER, index_dir=DOCS_INDEX_DIR)
    
    query_engine_tools = [
        QueryEngineTool(
            query_engine=system_docs_index.as_query_engine(similarity_top_k=15, child_branch_factor=2),
            metadata=ToolMetadata(
                name="designdocs",
                description="Documentation and design for the system",
            ),
        )
    ]

    texts = [
        f"Security controls and design of {SHORT_SYSTEM_DESC}, are in the design documentation and are provided. Extract all technical information about its design, data flows, encryption, other security and technical details as possible, and structure it to use for decision-making in threat modelling. Do not omit details about protocols, technologies, algorithms, business logic."
    ]
    context_docs = [Document(text=t) for t in texts]
    context_index = VectorStoreIndex.from_documents(context_docs)
    
    context_agent = ContextRetrieverOpenAIAgent.from_tools_and_retriever(
        query_engine_tools, context_index.as_retriever(similarity_top_k=2), 
        verbose=True,
        llm=LLM,
        qa_prompt='''
Context information is below.\n
---------------------\n
{context_str}\n
---------------------\n
Given the context information and your cybersecurity understanding and knowledge,
answer the function: {query_str}\n       
        ''',
        system_prompt=SYSTEM_PROMPT
    )

    return context_agent


def save_response(response):
    """
    helper function to save generated threat model in various formats 
    """
    with open("model_latest_run.md", "w") as file:
        file.write(response)
    print("[INFO] Model and details saved to 'model_latest_run.mmd'")
    
    response_lines = response.split("\n")
    start_index = -1
    end_index = -1

    for i, line in enumerate(response_lines):
        if line.startswith("```mermaid"):
            start_index = i
        elif line.startswith("```"):
            end_index = i

    if start_index != -1 and end_index != -1 and start_index < end_index:
        mermaid_lines = response_lines[start_index + 1 : end_index]

        with open("model_latest_run.mmd", "w") as file:
            file.write("\n".join(mermaid_lines))

        command = f"mmdc -i {os.path.join(SCRIPT_DIR, 'model_latest_run.mmd')} -o {os.path.join(SCRIPT_DIR, 'model_latest_run.svg')}"
        
        try:
            os.system(command)
            print("[INFO] Generated model as md, mmd and svg")
        except Exception as e:
            print(f"[ERR] Error running the command: {e}")
    else:
        print("[WARN] Generated response does not match the expected Mermaid format.")    

  
def core_loop():
    context_agent = prepare_agent()
    
    original_message = f"You're performing threat modeling and attack modeling on {SHORT_SYSTEM_DESC}. Your only task is to return Mermaid graph for how attackers can achieve at least 3 ATTACKER GOALS or more by exploiting weaknesses of the system by taking ATTACK STEPS. Each attack must consist of at least 5 or more ATTACK STEPS, start with Base((Attacker)) and end with one or more ATTACKER GOAL. Don't return ANYTHING else. Your response MUST start with '```mermaid' and end with '```'. Then, explain your thinking about vulnerabilities, threats and weaknesses you put in the Mermaid diagram. "
    
    context_agent.reset()
    response = str(context_agent.chat(original_message))

    print("[INFO] Mermaid threat model:")
    print(response)
    
    save_response(response)    

    # the chat loop
    while True:
        user_input = input("Enter a suggestion how to improve generated model (or 'exit' to quit): ")
        
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        
        original_message = f"You're performing threat modeling and attack modeling on {SHORT_SYSTEM_DESC}. Return ONLY Mermaid graph for at least 3 critical and impactful weaknesses of the system that can lead to devastating attacks on the system.\n{user_input}\n Don't return anything else first. Your response must start with '```mermaid' and end with '```'. Then, explain your thinking about vulnerabilities, threats and weaknesses you put in the Mermaid diagram."

        response = str(context_agent.chat(original_message))
        save_response(response)

if __name__ == "__main__":
    core_loop()
