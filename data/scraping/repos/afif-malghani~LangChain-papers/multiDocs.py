from langchain.document_loaders import PyPDFLoader
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from time import time_ns
from time import time


start = time_ns()

PAPERS_DIRECTORY = "./papers"
R_SPLITTER_CHUNK_SIZE = 450
R_SPLITTER_CHUNK_OVERLAP = 0
R_SPLITTER_SEPERATORS = ["\n\n", "\n", " " ,""]
ENCODER_MODEL = "all-MiniLM-L6-v2"
ENCODER_KWARGS = {'normalize_embeddings': False}
CHROMA_COLLECTOIN_NAME = "testCollection"

QUESTION = "Summerize the paper DocScanner: Robust Document Image Rectification with Progressive Learning"

LLM_MODEL = "meta-llama/Llama-2-7b-chat-hf"


loader = PyPDFDirectoryLoader(PAPERS_DIRECTORY)

docs = loader.load()

docsLoadTime = time_ns()

print("Documents loaded in "+ str(docsLoadTime - start) +" ns")


r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=R_SPLITTER_CHUNK_SIZE,
    chunk_overlap=R_SPLITTER_CHUNK_OVERLAP,
    separators=R_SPLITTER_SEPERATORS
)

split_docs = r_splitter.split_documents(docs)

splitTime = time_ns()

print("Documents spliting took " + str(splitTime - docsLoadTime) + " ns")


hf_emb = HuggingFaceEmbeddings(
    model_name=ENCODER_MODEL,
    encode_kwargs=ENCODER_KWARGS
)

embLoadTime = time_ns()

print("Embedding model loaded in "+ str(embLoadTime-splitTime)+" ns")

db = Chroma(CHROMA_COLLECTOIN_NAME, hf_emb)
db.add_documents(split_docs)
print("\n")
print(len(split_docs))
print("\n")

docs = db.similarity_search(QUESTION, k=1)
print(docs[0].page_content)


db.persist()


metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The research paper which the chunk belongs to",
            type="string"
            ),
        AttributeInfo(
            name="page",
            description="The page of the research paper from where the chunk was taken",
            type="integer"
            )
        ]

document_content_description = "Scientific research papers"

qaStart = time()

llm = HuggingFacePipeline.from_model_id(
    model_id=LLM_MODEL,
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 1000},
)

modelLoad = time()

print("Model load took "+str(modelLoad-qaStart)+" seconds")

retriever = SelfQueryRetriever.from_llm( llm, db, document_content_description, metadata_field_info, verbose=True)

print("retriever creation time: "+str(time()-modelLoad)+" seconds")

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())


questionTime = time()
result = qa_chain({"query": QUESTION})
print("time taken to answer "+ str(time()-questionTime) +" seconds")

print(result["result"])
