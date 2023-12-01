"""Ask a question to the notion database."""
import faiss
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
import pandas as pd
import dvc.api


params = dvc.api.params_show()
chat_params = params['ChatOpenAI']
qa_params = params['Retrieval']
print(chat_params)
print(qa_params)

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

with open("samples.txt", "r") as f:
    sample_questions = f.readlines()

store.index = index
llm = ChatOpenAI(temperature=chat_params['temperature'], model_name=chat_params['model_name'], max_retries=chat_params['max_retries'], verbose=chat_params['verbose'])
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=store.as_retriever(),
                                                    max_tokens_limit=qa_params['max_tokens_limit'],
                                                    reduce_k_below_max_tokens=qa_params['reduce_k_below_max_tokens'],
                                                    verbose=qa_params['verbose'])

records = []
for question in sample_questions:
    question = question.strip()
    print(f"Question: {question}")

    result = chain({"question": question})
    records.append({"Q": question, "A": {result["answer"].strip()}, "sources": result['sources'].strip()})

    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print("")

df = pd.DataFrame.from_records(records)
df.to_csv("results.csv", header=True, index=False)
