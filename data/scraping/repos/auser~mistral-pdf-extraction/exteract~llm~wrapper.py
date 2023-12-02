import box
import yaml, os
from exteract.paths import BASE_DIR

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from exteract.llm.prompts import qa_template
from exteract.llm.llm import setup_llm

# Import config vars
with open(BASE_DIR / "config.yml", "r", encoding="utf8") as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=qa_template, input_variables=["context", "question"]
    )
    return prompt


def build_retrieval_qa_chain(llm, prompt, vectordb):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": cfg.VECTOR_COUNT}),
        return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def setup_qa_chain():
    path = BASE_DIR / cfg.DB_FAISS_PATH

    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDINGS, model_kwargs={"device": "cpu"}
    )

    vectordb = FAISS.load_local(path, embeddings)
    llm = setup_llm()
    qa_prompt = set_qa_prompt()
    qa_chain = build_retrieval_qa_chain(llm, qa_prompt, vectordb)

    return qa_chain
