"""Module to define API routing and handle interactions with language models."""

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel # pylint: disable=E0611
from langchain.llms import OpenAI
from langchain.llms import VertexAI
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from src.hosted_llm import HostedLLM
from src.hosted_llm import CustomLlamaParser
from src.config import Config
from src.embeddings import EmbeddingSource
from src.logging_setup import setup_logger

config = Config()
logger = setup_logger()

router = APIRouter()

class HandleRequestPostBody(BaseModel): # pylint: disable=R0903
    """Class to define the request body for the handle_request_post endpoint."""
    user_input: str

def call_language_model(input_val):
    """Call the language model and return the result.

    Args:
        input_val: The input value to pass to the language model.

    Returns:
        The result from the language model.
    """
    model_provider = config.get("MODEL_PROVIDER", "UNDEFINED")
    logger.debug("Using model provider: %s", model_provider)
    prompt = PromptTemplate(
        input_variables=["input_val"],
        template="Pay close attention to the following... {input_val}",
    )
    if model_provider == 'openai':
        result = call_openai(input_val, prompt)
    elif model_provider == 'vertex':
        result = call_vertexai(input_val, prompt)
    elif model_provider == 'hosted':
        result = call_hosted_llm(input_val, prompt)
    else:
        raise ValueError(f"Invalid model name: {model_provider}")
    return result


def call_hosted_llm(input_val, prompt):
    """Call the hosted language model and return the result.

    Args:
        input_val: The input value to pass to the language model.

    Returns:
        The result from the language model.
    """
    hosted_model_name = config.get("HOSTED_MODEL_NAME", "Llama2-Hosted")
    logger.debug("Using self-hosted model: %s", hosted_model_name)
    hosted_model_uri = config.get("HOSTED_MODEL_URI", None)
    llm = HostedLLM(uri=hosted_model_uri)
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=CustomLlamaParser())
    result = chain.run(input_val)
    return result


def call_vertexai(input_val, prompt):
    """Call the Vertex AI language model and return the result.

    Args:
        input_val: The input value to pass to the language model.

    Returns:
        The result from the language model.
    """
    vertex_model_name = config.get("VERTEX_MODEL_NAME", "text-bison")
    logger.debug("Using Vertex AI model: %s", vertex_model_name)
    llm = VertexAI(model_name=vertex_model_name, temperature=float(
        config.get("MODEL_TEMPERATURE", 0.0)))
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input_val)
    return result

def call_openai(input_val, prompt):
    """Call the OpenAI language model and return the result.

    Args:
        input_val: The input value to pass to the language model.

    Returns:
        The result from the language model.
    """
    openai_model_name = config.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
    logger.debug("Using OpenAI model: %s", openai_model_name)
    if not spend_limit_exceeded():
        llm = OpenAI(model_name=openai_model_name, temperature=float(
            config.get("MODEL_TEMPERATURE", 0.0)))
        chain = LLMChain(llm=llm, prompt=prompt)
        with get_openai_callback() as openai_callback:
            result = chain.run(input_val)
        token_cost(openai_callback.total_tokens)
    else:
        raise HTTPException(status_code=402, detail="Spending limit exceeded")

    return result

def token_cost(total_tokens):
    """Calculate the cost of the tokens and log it.

    Args:
        total_tokens: The total number of tokens used.

    Returns:
        The total cost of the tokens.
    """
    spend_log_file = config.get("SPEND_LOG_FILE", "spend.log")
    openai_model_price = config.get("OPENAI_MODEL_PRICE", "0.000006")
    total_cost = total_tokens * float(openai_model_price)
    logger.debug("Total tokens: %f", total_tokens)
    logger.debug("Total cost: $%.5f", total_cost)

    with open(spend_log_file, "a", encoding="utf-8") as file:
        file.write(f"{total_cost:.5f}\n")
    total_spent = calculate_total_spent(spend_log_file)
    return JSONResponse(
        {
            "total_tokens": total_tokens,
            "total_cost": f"${total_tokens:.5f}",
            "total_spent": f"${total_spent:.5f}",
        }
    )

def calculate_total_spent(spend_log_file):
    """Calculate the total amount spent on tokens.

    Args:
        spend_log_file: The file containing the spend log.

    Returns:
        The total amount spent on tokens.
    """
    with open(spend_log_file, "r", encoding="utf-8") as file:
        total_spent = sum(float(line.strip()) for line in file)
    logger.debug("Total Spent: $%.5f", total_spent)
    return total_spent

def spend_limit_exceeded():
    """Check whether the spend limit has been exceeded.

    Args:
        None

    Returns:
        True if the spend limit has been exceeded, False otherwise.
    """
    spend_log_file = config.get("SPEND_LOG_FILE", "spend.log")
    spend_limit = config.get("SPEND_LIMIT", "0.001")
    total_spent = calculate_total_spent(spend_log_file)
    if total_spent > float(spend_limit):
        logger.error("SPEND_LIMIT ($%.5f) exceeded: $%.5f", float(spend_limit), total_spent)
        return True
    # use SPENDING_WARNING_PCT to determine when to send warning
    spending_warning_pct = config.get("SPENDING_WARNING_PCT", "0.8")
    if float(spend_limit) * float(spending_warning_pct) < total_spent <= float(spend_limit):
        logger.warning("SPEND_LIMIT warning")
    logger.debug("SPEND_LIMIT ($%.5f) not exceeded: $%.5f", float(spend_limit), total_spent)
    return False

def get_bot_response(user_input):
    """Get the bot response.

    Args:
        user_input: The user input to pass to the bot.

    Returns:
        The bot response.
    """
    logger.info(user_input)
    if user_input == 'hello':
        return 'Hi there!'
    if user_input == 'what is your name?':
        return 'My name is Chat Bot!'
    return call_language_model(user_input)


@router.get("/api_version_test/")
async def read_items():
    """Test Endpoint for API v1.

    Returns:
        list: The API current version.
    """
    return [{"version": "v1"}]


@router.post("/")
async def handle_request_post(request_body: HandleRequestPostBody):
    """Endpoint to handle POST requests for user input.

    Args:
        request_body: The body of the request containing user input.

    Returns:
        dict: A dictionary containing the bot response.
    """
    user_input = request_body.user_input
    bot_response = get_bot_response(user_input)
    return {"bot_response": bot_response}

@router.post("/find_sources", responses = {401: {
                                        "description": "PostgreSQL connection failed",
                                        "content": {
                                            "application/json": {
                                                "schema": {
                                                }
                                            }
                                        }
                                }})
def get_embedding_source(query: str = Body("step by step instructions to install a new operator"),
                        num_results: int = Body(3)):
    """Endpoint to get embedding sources for a given query.

    Args:
        request_body: The body of the request containing the query and number of results.

    Returns:
        dict: A dictionary containing the embedding source.
    """

    if isinstance(query, list):
        query = ' '.join(query)
    embeddings = EmbeddingSource()
    result = embeddings.get_source(query, num_results)
    return {"find_sources": result}


@router.post("/ask", responses = {402: {
                                        "description": "Payment Required",
                                        "content": {
                                            "application/json": {
                                                "schema": {
                                                }
                                            }
                                        }
                                }})
def synthesize_response(
                        query: str = Body("step by step instructions to install a new operator"),
                        num_results: int = Body(3),
                        prompt: str = Body(None)
                    ):
    """Endpoint to synthesize a response to a user query.

    Args:
        query: The user query.
        num_results: The number of results to return.
        prompt: The prompt to use for the response.

    Returns:
        dict: A dictionary containing the bot response.
    """
    if isinstance(query, list):
        query = ' '.join(query)

    embeddings = EmbeddingSource()
    embedding_results = embeddings.get_source(query, num_results)

    embedding_results_text = '\n\n---\n\n'.join([
        (
            f"Source: <a href=\"{result.get('source_link', '#')}\">"
            f"{result['source']}</a>\n\nContent:\n\n{result['content']}"
        )
        for result in embedding_results
    ])

    if prompt is None:
        prompt = (
            "<s>[INST] <<SYS>> \n"
            "Below is the only information you know.\n"
            "It was obtained by doing a vector search for the user's query:\n\n"
            "---START INFO---\n\n{embedding_results}\n\n"
            "---END INFO---\n\nYou must acknowledge the user's original query "
            f"of \"{query}\". "
            "Attempt to generate a summary of what you know from the sources provided "
            "based ONLY on the information given and ONLY if it relates to the original query. "
            "Use no other knowledge to respond. Do not make anything up. "
            "You can let the reader know if do not think you have enough information "
            "to respond to their query...\n\n"
            "<</SYS>>"
            f"{query} [/INST]"
        )

    prompt = prompt.format(embedding_results=embedding_results_text)
    logger.info("Query: %s", query)
    logger.info("Prompt: %s", prompt)
    bot_response = call_language_model(prompt)

    sources_used = [
        f"<a href=\"{result.get('source_link', '#')}\">{result['source']}</a>"
        for result in embedding_results
    ]
    if sources_used:
        bot_response += "\n\nPossibly Related Sources:\n" + '\n'.join(sources_used)
    else:
        bot_response += "\n\nNo Sources Found"

    return {"bot_response": bot_response}
