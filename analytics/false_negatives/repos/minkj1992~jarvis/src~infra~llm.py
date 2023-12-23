import asyncio
from enum import Enum, auto

import langchain
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chains import ConversationalRetrievalChain, FlareChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, VertexAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.base import VectorStore

from app.logger import get_logger
from infra.config import get_config
from infra.jbs4 import extract_doc_metadata_from_url

logger = get_logger(__name__)

_cfg = get_config()
_CHAT_OPEN_AI_TIMEOUT=240


async def get_docs_from_texts(texts:str):
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=_cfg.chunk_size, chunk_overlap=20, separators=["\n\n", "\n", " ", ""])
    for chunk in text_splitter.split_text(texts):
        docs.append(chunk)
    return docs


async def get_docs_and_metadatas_from_urls(urls):
    docs = []
    metadatas = []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=_cfg.chunk_size, chunk_overlap=20, separators=["\n\n", "\n", " ", ""])
    result = await asyncio.gather(
        *[extract_doc_metadata_from_url(url) for url in urls]
    )
    for (doc, metadata) in result:
        for chunk in text_splitter.split_text(doc):
            docs.append(chunk)
            metadatas.append(metadata)
    return docs, metadatas



async def get_chain(vs: VectorStore, prompt:str)-> ConversationalRetrievalChain:
    qa_prompt = PromptTemplate(template=prompt, input_variables=["context", "question"])
    return ConversationalRetrievalChain.from_llm(
        ChatOpenAI(
            openai_api_key=_cfg.openai_api_key, 
            temperature=_cfg.temperature, 
            model_name="gpt-3.5-turbo",
            request_timeout=_CHAT_OPEN_AI_TIMEOUT,
        ),
        retriever=vs.as_retriever(search_kwargs={'k':5}),
        qa_prompt=qa_prompt,
    )


condense_template = """Given the following conversation and a follow up question, do not rephrase the follow up question to be a standalone question. You should assume that the question is related to Chat history.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)



DEFAULT_PROMPT_TEMPLATE = """I want you to act as a document that I am having a conversation with. Your name is 'AI Assistant'. You will provide me with answers from the given info. If the answer is not included, say exactly '음... 잘 모르겠어요.' and stop after that. Refuse to answer any question not about the info. Never break character.

{context}

Question: {question}
!IMPORTANT Answer in korean:"""

async def get_chain_stream(vs: VectorStore, prompt:str, question_handler:AsyncCallbackHandler, stream_handler: AsyncCallbackHandler):
    manager = AsyncCallbackManager([])
    qa_prompt = PromptTemplate(template=prompt, input_variables=["context", "question"])

    question_generator = LLMChain(
        llm=ChatOpenAI(
            openai_api_key=_cfg.openai_api_key,
            temperature=_cfg.temperature,
            callback_manager=AsyncCallbackManager([question_handler]), 
            request_timeout=_CHAT_OPEN_AI_TIMEOUT,
            model_name=_cfg.qa_model,
            max_retries=3,
            ),
        prompt=CONDENSE_QUESTION_PROMPT,
        verbose=True,
    )
    
    streaming_llm = ChatOpenAI(
        streaming=True,
        temperature=_cfg.temperature,
        openai_api_key=_cfg.openai_api_key, 
        callback_manager=AsyncCallbackManager([stream_handler]), 
        request_timeout=_CHAT_OPEN_AI_TIMEOUT,
        model_name=_cfg.qa_model,
        verbose=True,
        max_retries=3,
    )
    
    doc_chain = load_qa_chain(
        streaming_llm,
        chain_type="stuff",
        prompt=qa_prompt,
        callback_manager=manager,
    )


    return ConversationalRetrievalChain(
        retriever=vs.as_retriever(search_kwargs={'k':4}),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
        max_tokens_limit=_cfg.max_token_limit,
        verbose=True
    )


class EmbeddingType(Enum):
    PALM = auto() # Google Cloud Platform Vertex AI PaLM
    OPENAI = auto() 

    


async def create_embeddings(et: EmbeddingType = EmbeddingType.OPENAI):
    """
    VertexAIEmbeddings Args:
        temperature: float = 0.0
            "Sampling temperature, it controls the degree of randomness in token selection."
        max_output_tokens: int = 128
            "Token limit determines the maximum amount of text output from one prompt."
        top_p: float = 0.95
            "Tokens are selected from most probable to least until the sum of their "
            "probabilities equals the top-p value."
        top_k: int = 40
            "How the model selects tokens for output, the next token is selected from "
            "among the top-k most probable tokens."
        project: Optional[str] = None
            "The default GCP project to use when making Vertex API calls."
        location: str = "us-central1"
            "The default location to use when making API calls."
        credentials: Any = None
            "The default custom credentials (google.auth.credentials.Credentials) to use "
            "when making API calls. If not provided, credentials will be ascertained from "
            "the environment."
    """
    if et == EmbeddingType.PALM:
        return VertexAIEmbeddings(
            temperature=_cfg.temperature,
            max_output_tokens=128,
            top_p=0.95,
            top_k=40,
        )
    return OpenAIEmbeddings(
            model=_cfg.embedding_model,
            openai_api_key=_cfg.openai_api_key,
            max_retries=3
    )


async def get_a_flare_chain(vs: VectorStore, prompt:str):
    # Compress
    llm = OpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=vs.as_retriever(search_kwargs={'k':4})
    )

    # TODO: use room prompt
    langchain.verbose = True
    # TODO: output with korean
    flare = FlareChain.from_llm(
        llm=ChatOpenAI(
            verbose=True,
            openai_api_key=_cfg.openai_api_key, 
            temperature=_cfg.temperature, 
            model_name="gpt-3.5-turbo",
            request_timeout=_CHAT_OPEN_AI_TIMEOUT,
        ),
        retriever=compression_retriever,
        max_generation_len=164,
        max_iter=4,
        min_prob=.3,
    )
    return flare



refine_prompt_template = (
    "The original question is as follows: {question}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question."
    "If the context isn't useful, return the original answer. Reply in Korean."
)
refine_prompt = PromptTemplate(
    input_variables=["question", "existing_answer", "context_str"],
    template=refine_prompt_template,
)


initial_qa_template = (
    "A chat conversation Context is below. The conversation format is 'year month day time, speaker: message'. For example, in '2000, May 3, 3:00 AM, A: Hello', the conversation content is Hello. The content of the conversation is the most important. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "You Must answer with reference to all your knowledge in addition to the information given\n"
    "!IMPORTANT Even if you can't analyze it, guess based on your knowledge. answer unconditionally.\n"
    "answer the question: {question}\nYour answer should be in Korean.\n"
)
initial_qa_prompt = PromptTemplate(
    input_variables=["context_str", "question"], template=initial_qa_template
)

from langchain.chains import RetrievalQAWithSourcesChain


async def get_a_refine_chain(vs: VectorStore, query:str):
    # refs: https://python.langchain.com/en/latest/modules/chains/index_examples/question_answering.html?highlight=refine#the-refine-chain
    llm = OpenAI(
        temperature=_cfg.temperature,
        openai_api_key=_cfg.openai_api_key, 
        request_timeout=_CHAT_OPEN_AI_TIMEOUT,
        model_name=_cfg.qa_model,
        verbose=True,
        max_retries=3,
    )

    qa_chain: RefineDocumentsChain = load_qa_with_sources_chain(
        llm=llm,
        chain_type="refine",
        return_refine_steps=True,
        question_prompt=initial_qa_prompt, 
        refine_prompt=refine_prompt,
    )
    
    qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain, 
        retriever=vs.as_retriever(search_kwargs={'k':2}))

    return qa(
        {"question": query}, 
        return_only_outputs=True,
    )

# TODO: from chatting to variable
summary_prompt_template = """Write a concise summary of the following chatting conversation in 3000 words:
    {docs}
CONCISE SUMMARY IN ENGLISH:
"""
async def get_a_summerize_report(vs: VectorStore, topic: str):
    # https://python.langchain.com/en/latest/modules/chains/index_examples/summarize.html
    retriever = vs.as_retriever(search_kwargs={'k':1}) # TODO: upgrade k to more than 7
    docs = retriever.get_relevant_documents(topic)
    llm = ChatOpenAI(
        temperature=_cfg.temperature,
        openai_api_key=_cfg.openai_api_key, 
        request_timeout=_CHAT_OPEN_AI_TIMEOUT,
        model_name=_cfg.qa_model,
        verbose=True,
        max_retries=3,
    )

    PROMPT = PromptTemplate(template=summary_prompt_template, input_variables=["docs"])
    chain = load_summarize_chain(
        llm=llm, 
        chain_type="map_reduce", # chain_type=refine
        combine_document_variable_name="docs",
        map_reduce_document_variable_name="docs",
        map_prompt=PROMPT, 
        combine_prompt=PROMPT
    )
    return chain({"input_documents": docs}, return_only_outputs=True)




llm_prompt_template = """Use the CONVERSATION CONTEXT below to write a 1500 ~ 2500 words report about the topic below.
    Determine the interset to be analyzed in detail with the TOPIC given below, and judge the flow of CONVERSATION CONTEXT based on the SUMMARY and interpret it according to the TOPIC.
    Create a report related to the TOPIC by referring to the CONVERSATION CONTEXT.
    The CONVERSATION CONTEXT format is 'year month day time, speaker: message'.
    
    For example, in 'A: Hello', the conversation content is Hello. 
    The content of the conversation is the most important.
    Please answer with reference to all your knowledge in addition to the information given by (TOPIC and SUMMARY and CONVERSATION CONTEXT). 
    
    !IMPORTANT Even if you can't analyze it, guess based on your knowledge. answer unconditionally.
    !IMPORTANT A REPORT must be in Korean.

    TOPIC: {topic}

    SUMMARY: {summary}
    
    CONVERSATION CONTEXT: {context}
    
    Answer in korean REPORT:"""

REPORT_PROMPT = PromptTemplate(
    template=llm_prompt_template, input_variables=["summary", "context", "topic"]
)

# https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_text_generation.html
async def get_a_relationship_report_from_llm(vs: VectorStore, topic: str, summary:str):
    llm = ChatOpenAI(
        temperature=_cfg.temperature,
        openai_api_key=_cfg.openai_api_key, 
        request_timeout=_CHAT_OPEN_AI_TIMEOUT,
        model_name=_cfg.qa_model,
        verbose=True,
        max_retries=3,
    )
    chain = LLMChain(llm=llm, prompt=REPORT_PROMPT)

    retriever = vs.as_retriever(search_kwargs={'k':2})
    docs = retriever.get_relevant_documents(topic)
    
    inputs = [{'summary':summary, 'context': doc.page_content, "topic": topic} for doc in docs]
    report = await chain.aapply(inputs)
        
    await logger.info(report)
    return report
    
