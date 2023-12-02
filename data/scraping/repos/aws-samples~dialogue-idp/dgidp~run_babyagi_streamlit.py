from babyagi import Optional, BabyAGI
from langchain import OpenAI
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import faiss, os


openai_api_key =  os.environ.get('openai_api_token')
# Define your embedding model
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

OBJECTIVE = "What happened to the First Republic Bank, another regional crisis in late April 2023? Will the FED take the same action as it did on SVB's failure?"

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
first_task = "Develop a task list"



# Logging of LLMChains
verbose = False
# If None, will keep on going forever
max_iterations: Optional[int] = 1
baby_agi = BabyAGI.from_llm_and_objectives(
                llm=llm,
                vectorstore=vectorstore,
                objective=OBJECTIVE,
                first_task=first_task,
                verbose=verbose
            )
baby_agi.run(max_iterations=max_iterations)
