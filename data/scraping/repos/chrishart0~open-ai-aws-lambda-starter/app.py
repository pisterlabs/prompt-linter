import logging
import logging.config
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain.chains import LLMChain


# Configure logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
    "loggers": {
        "uvicorn.error": {
            "level": "INFO",
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["console"],
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("uvicorn.info")

# Export logger as a function
def get_logger():
    """
    :return: Logger object
    Example of how to import
    from app import get_logger
    logger = get_logger()
    """
    return logger


logger.info("Initializing the things which need that to be prepared")

def get_openai_api_key():
    with open('openai_api_key.txt', 'r') as file:
        api_key = file.read().strip()
    return api_key
    

def call_llm(messages, api_key):
    """
    :param message: Message to send to OpenAI API
    :param api_key: OpenAI API key
    :return: Response from OpenAI API

    This function calls the OpenAI API with the provided message and returns the response.
    """

    # Prompt 
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a nice chatbot having a conversation with a human."
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    try:
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = api_key
        
        logger.info("Calling OpenAI API with Langchain") 
        chat = ChatOpenAI(model_name="gpt-3.5-turbo")

        # Convert messages to schema for Langchain
        chatHistory = []
        for messages in messages:
            if messages['role'] == 'user':
                chatHistory.append(HumanMessage(content=messages['content']))
            elif messages['role'] == 'assistant':
                chatHistory.append(SystemMessage(content=messages['content']))

        # Call OpenAI API with provided messages and model
        message = chat(chatHistory).content
        logger.info("Message: %s", message)


        return {'role': 'assistant', 'content':message}
    
    except Exception as e:
        # Log error if any exception occurs
        logger.error(e)
        logger.error(f"Error in calling OpenAI API: {str(e)}")
        # Propagate the exception further
        raise e
