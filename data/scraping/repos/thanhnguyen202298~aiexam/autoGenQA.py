from llama_index.evaluation import (
    DatasetGenerator,
    ResponseEvaluator,
    QueryResponseEvaluator,
)
from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    LLMPredictor,
    GPTVectorStoreIndex,
    load_index_from_storage,
    StorageContext,
)
from langchain.chat_models import ChatOpenAI
import os
import json
from core import config as apiSetting

keyAI = apiSetting.config["apiKey"]

os.environ["OPENAI_API_KEY"] = keyAI

# Load documents
reader = SimpleDirectoryReader(input_files=["raw_data/min_btc.csv"])
documents = reader.load_data()

# setup gpt model
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, chunk_size_limit=3000
)

# 1. Generate Questions
data_generator = DatasetGenerator.from_documents(
    documents, service_context=service_context
)
questions = data_generator.generate_questions_from_nodes()
# print(questions)

"""
print(questions)
Output:
["What is the main focus of Dirox Labs' services?", 'How can Artificial Intelligence and Machine Learning benefit businesses?', 'What are some sub-fields of Artificial Intelligence?', 'How does machine learning differ from deep learning?', 'What is the concept of Artificial Intelligence as a Service (AIaaS)?', 'How can A.I and ML improve organizational productivity?', 'What are the potential advantages of partnering with Dirox Labs?', 'How does A.I replicate human intelligence?', 'What percentage of leading businesses have invested in A.I technology?', 'How can A.I and Machine Learning be used to unlock potential markets and streamline operational processes?']
"""

# 2. For each question, generate an answer and then save a pair of Question-Answer to file /json/btc_llama.jsonl

# Create Index
index = GPTVectorStoreIndex.from_documents(documents)

# save index to disk
index.set_index_id("vector_index")
index.storage_context.persist("storage")

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="storage")
# load index
index = load_index_from_storage(storage_context, index_id="vector_index")

# Query the index
query_engine = index.as_query_engine(
    similarity_top_k=3, service_context=service_context
)

jsonl_file = "json/btc_llama.jsonl"
with open(jsonl_file, "w", encoding="utf-8-sig") as f:
    for question in questions:
        answer = query_engine.query(question)
        print(f"Question {question}, answer:{answer}")
        pair = json.dumps({"question": question, "answer": f"{answer}"})
        f.write(pair)
        f.write("\n")
