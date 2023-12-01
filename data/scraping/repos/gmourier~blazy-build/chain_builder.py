"""Chain builder"""
from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore

from langchain.prompts.prompt import PromptTemplate

def get_chat_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ChatVectorDBChain:
    condense_prompt = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate(
        template=condense_prompt, input_variables=["chat_history", "question"]
    )

    qa_prompt = """You are an AI assistant for the open source Meilisearch search engine.
You are given the following extracted parts of a long meilisearch document from the official documentation and a question. ONLY provide EXISTING links to the official documentation hosted at https://docs.meilisearch.com/. DO NOT try to make up link that DO NOT exist. Replace the .md extension by .html
You should only use links that are explicitly listed as a source in the context.
If the question includes a request for code, provide a code block directly from the documentation.
If you don't know the answer, just say "Hmm, I'm not sure." DO NOT try to make up an answer.
If you know the answer, remember that you are speaking to developers, so try to be as precise as possible.
If the question is not about Meilisearch, politely inform them that you are tuned to only answer questions about Meilisearch.
If you know the answer, DO NOT include cutted parts.
DO NOT start the answer with <br> tags.
QUESTION: {question}
=========
{context}
=========
MARKDOWN ANSWER:"""

    QA_PROMPT = PromptTemplate(
        template=qa_prompt, input_variables=["context", "question"]
    )

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
        temperature=0.2,
        max_tokens=500
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    )

    chat = ChatVectorDBChain(
        vectorstore=vectorstore,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
    )
    return chat

def get_qa_chain(
    vectorstore: VectorStore
) -> VectorDBQAWithSourcesChain:
    qa_prompt = """Given the following extracted parts of a long documentation and a question, create a final answer with references ("SOURCES").
If you don't know the answer, DO NOT try to make up an answer and DO NOT include any sources, just say that you are sorry and you don't know, add a programmer joke.
If you know the answer, remember that you are speaking to developers, so try to be as precise as possible.
If you know the answer, return a "SOURCES" array in your answer, never write it "SOURCE" and indicate the relevance of each source with a "SCORE" between 0 and 1, only return sources with a score superior to 0.8, rank them by their score.
Return the "SOURCES" array with the following format: url: url, score: score.
If you know the answer, DO NOT include cutted parts.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

    PROMPT_STUFF = PromptTemplate(template=qa_prompt, input_variables=["summaries", "question"])

    """Create a VectorDBQAWithSourcesChain for question/answering."""
    llm = OpenAI(
        temperature=0.2,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=PROMPT_STUFF)

    qa = VectorDBQAWithSourcesChain(
        vectorstore=vectorstore,
        combine_documents_chain=doc_chain,
    )
    return qa