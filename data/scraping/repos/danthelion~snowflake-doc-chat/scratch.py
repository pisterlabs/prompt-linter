import openai

import chromadb
from chromadb import Settings
from chromadb.utils import embedding_functions
import langchain
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate


from utils.constants import VECTOR_DB_DIR

chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=VECTOR_DB_DIR,
))

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key="sk-",
                model_name="text-embedding-ada-002"
            )

coll = chroma_client.get_collection("langchain", embedding_function=openai_ef)

dox = coll.query(query_texts="Users and Roles", n_results=8)

_prompt = """
You are a quiz asking machine, helping people prepare for exams about a certain topic. Based on the information below,
construct a quiz with 5 questions and answers. The questions should be single choice questions with 4 possible answers.
The answers should be the correct answer and 3 distractors. Come up with 3 questions in total.

Example 1:

Question: In which layer of its architecture does Snowflake store its metadata statistics?

A. Storage Layer
B. Compute Layer
C. Database Layer
D. Cloud Services Layer

Answer: D

Example 2:

Question: Which type of table corresponds to a single Snowflake session?

A. Temporary
B. Transient
C. Provisional
D. Permanent

Answer: A

Context to generate questions from:
{context}
"""

prompt = PromptTemplate.from_template(_prompt)

llm = OpenAI(temperature=0.0, openai_api_key="s)


chain = LLMChain(llm=llm, prompt=prompt)
res = chain.run("\n".join([d for d in dox['documents'][0]]))

print(res)