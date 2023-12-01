from chat_with_x.utils.processing_utils import file_helper_prompt
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from chat_with_x.utils.callbacks import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.base import AsyncCallbackManager
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ChatVectorDBChain
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.base import CallbackManager
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain

# from langchain.chains import ChatVectorDBChain


def get_prompts(one_line_file_desc, task_type="initial", instruction="conversational"):
    """ Returns the appropriate prompt template for the given task type and instruction

    Args:
        one_line_file_desc (str): A one line description of the file to be used in the prompt
        task_type (str): The type of task to be performed. Can be "initial" or "follow_up"
        instruction (str): The type of instruction to be used in the prompt. Can be "conversational" or "standalone"

    Returns:
        The appropriate prompt templates
    """
    _template = "Given the following conversation and a follow up user input, rephrase the follow up input to be a " \
                f"standalone input (often a question). You can assume the user input is about {one_line_file_desc}. " \
                "\nChat History:\n{chat_history}\nFollow Up Input:\n{question}\nStandalone input:"
    template = file_helper_prompt(one_line_file_desc, task_type=task_type, instruction=instruction)

    _condense_prompt = PromptTemplate.from_template(_template)
    _qa_prompt = PromptTemplate(template=template, input_variables=["question", "context"])
    return _qa_prompt, _condense_prompt


def get_chain(vectorstore, one_line_file_desc, search_type="similarity",
              task_type="initial", instruction="conversational",
              model_name="gpt-3.5-turbo", model_temperature=0.0):
    """ TBD """

    chat_model = ChatOpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                            verbose=True, model_name=model_name, temperature=model_temperature)

    # Retrieve prompts
    qa_prompt, condense_prompt = get_prompts(one_line_file_desc, task_type, instruction)

    # Build chain and return
    # _chain = ChatVectorDBChain.from_llm(
    #     llm, vectorstore, qa_prompt=qa_prompt, condense_question_prompt=condense_prompt
    # )
    _chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(search_type=search_type),
        qa_prompt=qa_prompt,
        condense_question_prompt=condense_prompt
    )
    return _chain


def get_chain_v2(vectorstore, question_handler, stream_handler,
                 one_line_file_desc, search_type="similarity",
                 task_type="initial", instruction="conversational",
                 model_name="gpt-3.5-turbo", model_temperature=0.0,
                 tracing=True):
    """Create a ChatVectorDBChain for question/answering."""

    # Get prompts
    qa_prompt, condense_prompt = get_prompts(one_line_file_desc, task_type, instruction)

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

    question_gen_llm = ChatOpenAI(
        model_name=model_name,
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = ChatOpenAI(
        model_name=model_name,
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=model_temperature,
    )

    question_generator = LLMChain(
        llm=question_gen_llm,
        prompt=condense_prompt,
        callback_manager=manager,
        verbose=True,
    )
    doc_chain = load_qa_chain(
        streaming_llm,
        chain_type="stuff",
        prompt=qa_prompt,
        callback_manager=manager,
        verbose=True,
    )
    _chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(search_type=search_type),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
    )
    return _chain
