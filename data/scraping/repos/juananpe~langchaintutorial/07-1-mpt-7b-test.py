from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from ctransformers.langchain import CTransformers
from langchain.embeddings import HuggingFaceInstructEmbeddings

llm = CTransformers(model='/tmp/mpt-7b-instruct.ggmlv3.q5_0.bin', 
                    model_type='mpt')

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                                      model_kwargs={"device": "cpu"})

db = FAISS.load_local("faiss_index", instructor_embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                  chain_type="stuff", 
                                  retriever=retriever)
<<<<<<< HEAD


# Running LLMs locally
# https://wandb.ai/capecape/LLMs/reports/A-Guide-on-Running-LLMs-Locally--Vmlldzo0Njg5NzMx
# Using NEW MPT-7B in Hugging Face and LangChain
# https://www.youtube.com/watch?v=DXpk9K7DgMo&ab_channel=JamesBriggs
# Running mpt-30b with a CPU
# https://github.com/abacaj/mpt-30B-inference
=======
>>>>>>> 5b078fd (add local LLM example)
