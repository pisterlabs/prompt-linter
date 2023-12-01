import os
import logging
import chainlit as cl
from chainlit.input_widget import Select, Slider

from langchain import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers, OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

logging.basicConfig(
    filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

llm_path = os.environ["LLM_PATH"]

local_models = [
    "",
    "codellama-7b.Q8_0.gguf",
    "codellama-13b.Q4_K_M.gguf",
    "codellama-34b.Q4_K_M.gguf",
    "llama-2-13b-chat.ggmlv3.q4_K_M.bin",
    "wizardcoder-python-34b-v1.0.Q3_K_M.gguf",
    "wizardcoder-python-34b-v1.0.Q3_K_L.gguf",
]

sentence_transformer_model = "all-MiniLM-L6-v2"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SERPAPI_API_KEY = os.environ["SERPAPI_API_KEY"]
DB_FAISS_PATH = os.environ["DB_FAISS_PATH"]

template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """
    This function sets up a custom prompt template for the language model.
    
    Returns:
        prompt: A custom prompt template object
    Raises:
        Exception: If any error occurs during the setup of the prompt template.
    """
    try:
        prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
        return prompt
    except Exception as e:
        logging.exception("Error occurred in set_custom_prompt function: ", exc_info=True)
        raise


def load_llm(local_model, openai_model, temperature):
    """
    This function loads the language model either from the local machine or from OpenAI's API.
    
    Args:
        local_model (str): The name of the local model to load.
        openai_model (str): The name of the OpenAI model to load.
        temperature (float): The temperature setting for the language model.
    
    Returns:
        llm: The loaded language model object.
    Raises:
        Exception: If any error occurs during the loading of the language model.
    """
    try:
        if local_model:
            llm = CTransformers(
                model=llm_path,
                model_name=local_model,
                n_batch=4096,
                n_ctx=4096,
                max_new_tokens=2048,
                temperature=temperature,
                callbacks=[StreamingStdOutCallbackHandler()],
                verbose=True,
                streaming=True,
            )

        if openai_model:
            llm = OpenAI(model_name=openai_model, api_key=OPENAI_API_KEY, temperature=temperature, max_tokens=2000)

        return llm

    except Exception as e:
        logging.exception("Error occurred in load_llm function: ", exc_info=True)
        raise


def qa_bot(llm):
    """
    This function sets up the question and answer bot with the loaded language model.
    
    Args:
        llm: The loaded language model object.
    
    Returns:
        qa: The initialized question and answer bot object.
    """
    embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{sentence_transformer_model}", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    qa_prompt = set_custom_prompt()
    qa = RetrievalQA.from_chain_type(
        llm=llm, retriever=db.as_retriever(search_kwargs={'k': 2}), return_source_documents=True, chain_type_kwargs={'prompt': qa_prompt}
    )
    return qa


@cl.on_chat_start
async def start():
    """
    This asynchronous function initiates the chat start process, setting up the chat settings, loading the model, initializing the agent and sending the starting message.
    
    Raises:
        Exception: If any error occurs during the chat start process.
    """
    try:
        settings = await cl.ChatSettings(
            [
                Select(id="OpenAIModel", label="OpenAI - Model", values=["", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"], initial_index=0),
                Select(id="LocalModel", label="Local - Model", values=local_models, initial_index=6),
                Slider(id="Temperature", label="Temperature", initial=0.5, min=0, max=2, step=0.1),
            ]
        ).send()
        # setup_agent(settings)
        local_model = settings["LocalModel"]
        openai_model = settings["OpenAIModel"]
        temperature = settings["Temperature"]
        llm = load_llm(local_model, openai_model, temperature)
        chain = qa_bot(llm)

        search = SerpAPIWrapper()
        llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

        tools = [
            Tool(name="Search", func=search.run, description="useful for when you need to answer questions about current events. You should ask targeted questions"),
            Tool(name="Calculator", func=llm_math_chain.run, description="useful for when you need to answer questions about math"),
        ]
        agent = initialize_agent(tools, llm, agent="chat-zero-shot-react-description", verbose=True)
        cl.user_session.set("agent", agent)

        msg = cl.Message(content="Starting the bot...")
        await msg.send()
        msg.content = "Hi, what would you like to ask?"
        await msg.update()
        cl.user_session.set("chain", chain)

    except Exception as e:
        logging.exception("Error occurred in start function: ", exc_info=True)
        raise


@cl.on_settings_update
async def setup_agent(settings):
    """
    This asynchronous function handles settings update during the chat session.
    
    Args:
        settings: The settings to update.
    Raises:
        Exception: If any error occurs during the settings update process.
    """
    try:
        print("on_settings_update", settings)
    except Exception as e:
        logging.exception("Error occurred in setup_agent function: ", exc_info=True)
        raise


@cl.on_message
async def main(message):
    """
    This is the main asynchronous function to handle incoming messages during the chat session, updating the agent and returning the response message.
    
    Args:
        message: The incoming message from the user.
    Returns:
        response: The response message from the bot.
    Raises:
        Exception: If any error occurs during the message handling process.
    """
    try:
        agent = cl.user_session.get("agent")  # type: AgentExecutor
        chain = cl.user_session.get("chain") # type: RetrievalQA
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True
        )
        cb.answer_reached = True
        res = await chain.acall(message, callbacks=[cb])
        # TODO: make the agents work
        # res = await cl.make_async(agent.run)(message, callbacks=[cb])
        # answer = res["result"]
        sources = res["source_documents"]
        main_message = await cl.Message(content="").send()

        if sources:
            sources_text = f"\n\n# Sources: \n"
            for source in sources:
                sources_text += '\n' + str(source.metadata['source'])
                sources_text += '\n\n'
                sources_text += source.page_content
        else:
            sources_text += "\n\nNo sources found"
        
        await cl.Message(content=sources_text, parent_id= main_message, author="Source: ").send()
    
    except Exception as e:
        logging.exception("Error occurred in main function: ", exc_info=True)
        raise


