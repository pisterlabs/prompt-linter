"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQAWithSourcesChain
from sentence_transformers import SentenceTransformer
from transformers import pipeline


def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ConversationalRetrievalChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    
    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = OpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )
    
    # question_gen_model_name = 'bert-base-nli-mean-tokens'  # Choose the appropriate sentence-transformers model
    # question_gen_llm = SentenceTransformer(question_gen_model_name)
    
    # streaming_model_name = 'distilbert-base-nli-stsb-mean-tokens'  # Choose the appropriate sentence-transformers model
    # streaming_llm = SentenceTransformer(streaming_model_name)

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    )
    
    qa = ConversationalRetrievalChain(
        # vectorstore=vectorstore,
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
        return_source_documents=True,
    )
    return qa



# question_generator = pipeline("text-generation", model="distilbert-base-uncased")
#     qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased")

#     doc_chain = load_qa_chain(
#         qa_pipeline, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
#     )

    # question_gen_model_name = 'bert-base-nli-mean-tokens'  # Choose the appropriate sentence-transformers model
    # question_gen_llm = SentenceTransformer(question_gen_model_name)
    
    # streaming_model_name = 'distilbert-base-nli-stsb-mean-tokens'  # Choose the appropriate sentence-transformers model
    # streaming_llm = SentenceTransformer(streaming_model_name)