"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.manager import AsyncCallbackManager

from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain


from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate

from langchain.chains.question_answering import load_qa_chain

from langchain.vectorstores.base import VectorStore

import os
from dotenv import load_dotenv


def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ChatVectorDBChain:
    load_dotenv()
    
    os.environ['OPEN_API_TYPE'] = "azure"
    openai_api_base = os.environ['OPENAI_API_BASE']
    openai_api_key = os.environ['OPENAI_API_KEY']
    azure_development_name = os.environ['AZURE_DEVELOPMENT_NAME']

    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])

    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    ) 

    
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    # use azure llm
    question_gen_llm = AzureOpenAI(
        deployment_name=azure_development_name, 
        model_name="gpt-35-turbo",
        temperature=0,
        verbose=True,
        callback_manager=question_manager
    )
    streaming_llm = AzureOpenAI(
        deployment_name=azure_development_name, 
        model_name="gpt-35-turbo",
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0
    )


    # ? update temperature  
    # question_gen_llm = OpenAI(
    #     openai_api_key=openai_key,
    #     temperature=0,
    #     verbose=True,
    #     callback_manager=question_manager,
    # )
    # streaming_llm = OpenAI(
    #     openai_api_key=openai_key,
    #     streaming=True,
    #     callback_manager=stream_manager,
    #     verbose=True,
    #     temperature=0,
    # )

    # azure chat
    # model = AzureChatOpenAI(
    #     openai_api_base=openai_api_base,
    #     openai_api_version="2023-03-15-preview",
    #     deployment_name=azure_development_name,
    #     openai_api_key=openai_api_key,
    #     openai_api_type='azure'
    # )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    )

    qa = ChatVectorDBChain(
        vectorstore=vectorstore,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
    )
    return qa
