"""Create a ChatVectorDBChain for question/answering."""
import os
os.environ["LANGCHAIN_HANDLER"] = "langchain"

from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore
from pydantic import Field

doc_template="""--- document start ---
href: {href}
section: {section}
content:{page_content}
--- document end ---
"""

QUARTO_DOC_PROMPT = PromptTemplate(
    template=doc_template,
    input_variables=["page_content", "href", "section"]
)

prompt_template = """You are an AI assistant for the open source library Quarto. The documentation is located at https://quarto.org/docs
You are given the following extracted parts of a long document and a question. Provide a conversational answer with a hyperlink to the documentation.
You can construct the hyperlink by using the href and section fields in the context and the base url https://quarto.org/.
You should only use hyperlinks that are explicitly listed as a source in the context. Do NOT make up a hyperlink that is not listed.
You should only show code examples that are explicitly listed in the documentation.  Do not make up code examples.
If the question includes a request for code, provide a fenced code block directly from the documentation.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about Quarto, politely inform them that you are tuned to only answer questions about Quarto.

Question: {question}

Documents:
=========
{context}
=========

Answer in Markdown:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ChatVectorDBChain:
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
        model_name='gpt-3.5-turbo',
        max_retries=15,
        max_tokens=520,
        temperature=0.5,
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = OpenAI(
        streaming=True,
        max_retries=15,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, document_prompt=QUARTO_DOC_PROMPT,
        callback_manager=manager
    )

    qa = ChatVectorDBChain(
        vectorstore=vectorstore,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
        top_k_docs_for_context=10
    )
    return qa
