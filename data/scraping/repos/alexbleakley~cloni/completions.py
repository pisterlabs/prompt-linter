import datetime
import logging
import os
import sys
import time

import ai21
import anthropic
import cohere
import dotenv
import openai


date_str = datetime.datetime.utcnow().strftime('%Y-%m-%d')
timestamp_str = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
log_file_path = f"logs/{date_str}/{timestamp_str}.log"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(filename=f"{log_file_path}", filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

dotenv.load_dotenv()
AI21_API_KEY = os.getenv('AI21_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

ai21.api_key = AI21_API_KEY
openai.api_key = OPENAI_API_KEY


def generate_ai21_completion(
    prompt: str,
    model: str,
    temperature: int = 0.7,
    max_tokens: int = 16,
) -> str:
    """Send prompt to the AI21 completion API and return the completion."""
    logging.debug(
        "Generating completion. "
        f"model={model}, temperature={temperature}, maxTokens={max_tokens}."
    )
    logging.debug(f"Request prompt:\n{prompt}")
    attempt_count = 0
    complete = False
    while not complete:
        try:
            response = ai21.Completion.execute(
                model=model,
                prompt=prompt,
                temperature=temperature,
                maxTokens=max_tokens,
            )
            complete = True
        except (
            ai21.errors.APIError,
            ai21.errors.UnprocessableEntity,
            ai21.errors.TooManyRequests,
            ai21.errors.ServerError,
            ai21.errors.ServiceUnavailable,
            ai21.errors.AI21ClientException,
        ):
            attempt_count += 1
            if attempt_count < 20:
                backoff_time = attempt_count ** 2
                logging.warning(f"Received AI21 error. Will retry after {backoff_time}s.")
                time.sleep(backoff_time)
                logging.info(f"Retrying...")
            else:
                logging.error(f"Received AI21 error {attempt_count} times. Aborting completion.")
                raise
    logging.debug(f"Response:\n{response}")
    response_text = response.completions[0].data.text
    return response_text


def generate_anthropic_completion(
    prompt: str,
    model: str, 
    temperature: int = 1, 
    max_tokens: int = 256,    
) -> str:
    """Send prompt to the Anthropic completions API and return the completion.
    
    The prompt argument will be converted into the Human/Assistant format required by Anthropic.
    """
    logging.debug(
        "Generating completion. " 
        f"model={model}, temperature={temperature}, max_tokens_to_sample={max_tokens}."
    )   
    logging.debug(f"Request prompt:\n{prompt}")
    anthropic_prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
    attempt_count = 0
    complete = False
    while not complete:
        try: 
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.completions.create(
                model=model,
                prompt=anthropic_prompt,
                temperature=temperature,
                max_tokens_to_sample=max_tokens,
            )
            complete = True
        except (
            anthropic.APIError,
            anthropic.APIConnectionError,
            anthropic.APIResponseValidationError,
            anthropic.APIStatusError,
            anthropic.APITimeoutError,
            anthropic.ConflictError,
            anthropic.InternalServerError,
            anthropic.NotFoundError,
            anthropic.RateLimitError,
            anthropic.UnprocessableEntityError,
        ):
            attempt_count += 1
            if attempt_count < 40:
                backoff_time = attempt_count ** 2
                logging.warning(f"Received Anthropic error. Will retry after {backoff_time}s.")
                time.sleep(backoff_time)
                logging.info(f"Retrying...")
            else:
                logging.error(f"Received Anthropic error {attempt_count} times. Aborting completion.")
                raise
    logging.debug(f"Response:\n{response}")
    response_text = response.completion
    return response_text  


def generate_cohere_completion(
    prompt: str,
    model: str, 
    temperature: int = 0.75, 
    max_tokens: int = 20,    
) -> str:
    """Send prompt to the Cohere generate API and return the completion.
    
    To avoid hitting rate limits, this function will wait 12 seconds before sending the request.
    """
    logging.debug(f"Waiting 12 seconds (Cohere trial rate limit = 5 calls/minute)...")
    time.sleep(12)
    logging.debug(
        "Generating completion. " 
        f"model={model}, temperature={temperature}, max_tokens={max_tokens}."
    )   
    logging.debug(f"Request prompt:\n{prompt}")
    attempt_count = 0
    complete = False
    while not complete:
        try: 
            client = cohere.Client(api_key=COHERE_API_KEY)
            response = client.generate(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            complete = True
        except (
            cohere.CohereError,
            cohere.CohereAPIError,
            cohere.CohereConnectionError,
        ):
            attempt_count += 1
            if attempt_count < 20:
                backoff_time = max(attempt_count ** 2, 12)
                logging.warning(f"Received Cohere error. Will retry after {backoff_time}s.")
                time.sleep(backoff_time)
                logging.info(f"Retrying...")
            else:
                logging.error(f"Received Cohere error {attempt_count} times. Aborting completion.")
                raise
    logging.debug(f"Response:\n{response}")
    response_text = response[0].text
    return response_text  


def generate_openai_completion(
    prompt: str,
    model: str, 
    temperature: int = 1, 
    max_tokens: int = 16,    
) -> str:
    """Send prompt to one of the OpenAI completion APIs and return the completion.
    
    Uses the legacy Completion API or the newer ChatCompletion API according to the model.
    If using the ChatCompletion API, the prompt argument will be converted into a list of messages 
    and the response will be extracted from the response message.
    """
    logging.debug(
        "Generating completion. " 
        f"model={model}, temperature={temperature}, max_tokens={max_tokens}."
    )   
    logging.debug(f"Request prompt:\n{prompt}")
    if model in [
        "ada", "babbage", "curie", "davinci", 
        "text-ada-001", "text-babbage-001", "text-curie-001", "text-davinci-001", "text-davinci-002", "text-davinci-003"
    ]:
        legacy = True
    else:
        legacy = False
        messages = [
            {
                'role': 'user', 
                'content': prompt,
            }
        ]
    attempt_count = 0
    complete = False
    while not complete:
        try: 
            if legacy:
                response = openai.Completion.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            complete = True
        except (
            openai.error.RateLimitError,
            openai.error.APIError, 
            openai.error.Timeout,
            openai.error.APIConnectionError,
            openai.error.ServiceUnavailableError,
        ):
            attempt_count += 1
            if attempt_count < 20:
                backoff_time = attempt_count ** 2
                logging.warning(f"Received OpenAI error. Will retry after {backoff_time}s.")
                time.sleep(backoff_time)
                logging.info(f"Retrying...")
            else:
                logging.error(f"Received OpenAI error {attempt_count} times. Aborting completion.")
                raise
    logging.debug(f"Response:\n{response}")
    if legacy:
        response_text = response['choices'][0]['text']
    else:
        response_text = response['choices'][0]['message']['content']
    return response_text    


def generate_completion(prompt: str, provider: str, model: str, **kwargs) -> str:
    """Send prompt to the specified provider's completion API and return the completion."""
    if provider == "openai":
        return generate_openai_completion(prompt, model, **kwargs)
    elif provider == "anthropic":
        return generate_anthropic_completion(prompt, model, **kwargs)
    elif provider == "cohere":
        return generate_cohere_completion(prompt, model, **kwargs)
    elif provider == "ai21":
        return generate_ai21_completion(prompt, model, **kwargs)
    else:
        raise NotImplementedError(f"Provider {provider} is not supported.")
   