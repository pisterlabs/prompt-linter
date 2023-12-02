import timeit
from loguru import logger
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import ai_driver.vector_storage.faiss_manager as FAISSManager
from ai_driver.local_llm.prompts import qa_template
from ai_driver.local_llm.ggml_llm import build_ggml_llm, GGMLConfig
from ai_driver.retrieval.qa import QADBConfig, qa_pipeline


def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=qa_template, input_variables=["context", "question"]
    )
    return prompt


def build_retrieval_qa(llm, prompt, vectordb, config: QADBConfig):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": config.vector_count}),
        return_source_documents=config.return_source,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def local_llm_qa_pipeline(
    query: str, qa_config: QADBConfig, ggml_config: GGMLConfig, device: str = "cuda"
):
    """Example Local LLM Pipeline"""

    # Setup DBQA
    start = timeit.default_timer()
    embeddings = FAISSManager.get_cache_embeddings(
        qa_config.embed_model, {"device": "cuda"}
    )
    vectordb = FAISSManager.get_vector_store(embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": qa_config.vector_count})
    model = build_ggml_llm(ggml_config)
    response = qa_pipeline(query, retriever, model)

    end = timeit.default_timer()

    logger.info(f"\nAnswer: {response}")
    logger.info("=" * 50)

    logger.info(f"Time to retrieve response: {end - start}")
    return response
