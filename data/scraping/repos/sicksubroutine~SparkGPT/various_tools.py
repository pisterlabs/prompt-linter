from flask import g, Request
from datetime import datetime as dt
from __init__ import debug_logger, logger
import string
import random
import uuid 
import hashlib 
import os
import openai 
import time 
import requests

API_KEY = os.environ['LNBITS_API']
URL = "https://legend.lnbits.com/api/v1/payments/"
HEADERS = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
SATS = 0.00000001
SECRETKEY = os.environ['GPT_API']
openai.api_key = f"{SECRETKEY}"


class DataUtils:

  @staticmethod
  def api_request(method: str, url: str, **kwargs: dict) -> tuple:
    """
    Sends an API request to the given URL using the given HTTP method 
    and keyword arguments.

    Args:
      method (str): The HTTP method to be used for the request.
      url (str): The URL for the request.
      **kwargs (dict): The keyword arguments to be passed to the request.

    Raises:
      Exception: If the API request fails.

    Returns:
      tuple: A tuple containing the response and the response JSON.
      The tuple has the following structure: (response, response_json)
    """
    try:
      response = requests.request(method, url, **kwargs)
      if not response.ok:
        raise Exception(
          f"API request failed with status code {response.status_code}.")
      response_json = response.json()
      return response, response_json
    except Exception as e:
      logger.error(f"API request failed: {e}")
      return None, None

  @staticmethod
  def time_get() -> str:
    time = dt.now()
    string_time = time.strftime("%m-%d-%Y %I:%M:%S %p")
    return string_time
  
  @staticmethod
  def saltGet() -> str:
    """
    Generate a random salt consisting of uppercase letters, lowercase letters,
    digits, and punctuation. The salt is 30 characters long.
    
    Returns:
    str: A random salt.
    """
    return ''.join(
      random.choice(string.ascii_letters + string.digits + string.punctuation)
      for _ in range(30))

  @staticmethod
  def tokenGet30() -> str:
    """
    Generate a random token consisting of uppercase letters and digits.
    The token is 30 characters long.

    Returns:
    str: A random token.
    """
    return ''.join(
      random.choice(string.ascii_letters + string.digits) for _ in range(30))

  @staticmethod
  def tokenGet16() -> str:
    """
    Generate a random token consisting of uppercase letters and digits.
    The token is 16 characters long.

    Returns:
    str: A random token.
    """
    ran_token = ''.join(
      random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
    return ran_token

  @staticmethod
  def get_IP_Address(request: Request) -> str:
    """
    Retrieves the IP address of the user.

    Args:
      request (flask.request): The request object.

    Returns:
      str: The IP address of the user.

    Notes:
      - The IP address is retrieved from the 'X-Forwarded-For' header.
      - If the IP address cannot be retrieved, None is returned.
    """
    try:
      return request.headers.get('X-Forwarded-For', '').split(',')[0].strip()
    except Exception as e:
      logger.error(f"Failed to get IP Address: {e}")
      return "Erroring getting IP Address"

  @staticmethod
  def uuid_func() -> str:
    """
    Generates a UUID or device_ID.

    Returns:
      str: A UUID.

    Notes:
      - The UUID is generated using the uuid1 method.
    """
    return f"{uuid.uuid1()}"

  @staticmethod
  def hash_func(*args) -> str:
    """
    Generates a hash based on the given arguments.

    Args:
      *args (str): The arguments to be hashed.

    Returns:
      str: A hash.

    Notes:
      - The arguments are concatenated and hashed using the SHA256 algorithm.
      - The hash is returned as a hexadecimal string.
    """
    a = ''.join(args)
    hash_str = hashlib.sha256(a.encode()).hexdigest()
    return hash_str

  @staticmethod
  def check_old_markdown() -> None:
    """
    Checks and removes outdated Markdown files from the 'static/markdown/' directory.

    This method checks if the 'static/markdown/' directory exists. 
    If it doesn't, the directory is created.
    Then, it iterates through each file in the directory 
    and removes any file with the '.md' extension.
    """
    path = "static/markdown/"
    if not os.path.exists(path):
      os.makedirs(path)
    for filename in os.listdir(path):
      if filename.endswith(".md"):
        os.remove(path + filename)

  @staticmethod
  def clean_up_invoices() -> None:
    """
    Cleans up old invoice QR code files from the 'static/qr/' directory.

    This method iterates through each file in the 'static/qr/' directory and 
    removes any file with the '.png' extension.
    """
    path = "static/qr/"
    for filename in os.listdir(path):
      if filename.endswith(".png"):
        os.remove(path + filename)

  @staticmethod
  def summary_of_messages(message: str) -> tuple:
    """
    Generates a summary of the user's messages in a conversation 
    and obtains a concise response.

    Args:
        message (str): The message to be summarized.

    Returns:
      tuple: A tuple containing the longer response and the concise response.
      The tuple has the following structure: (longer_response, concise_response)

    Notes:
      - Takes the first message from the user and generates a summary.
      - A prompt is created with a specific format for summarization.
      - The user's summary messages are included in the 
      conversation along with the system prompt.
      - The original longer response is preserved before any modifications.
      - The concise response is generated by removing punctuation and 
      joining the response words with underscores.
    """
    
    prompt = """The following message should be summarized as your output.
          Output should have no explanation or elaboration. Just a summary.
          Output is required to be seven words or less with no punctuation."""
    summary = [{
      "role": "system",
      "content": f"{prompt}"
    }, {
      "role": "user",
      "content": message
    }]
    response, _ = ChatUtils.openai_response(summary, "gpt-3.5-turbo")
    longer_response = response
    response = response.split()
    response = "_".join(response)

    response = response.replace(".", "")
    response = response.replace(",", "")
    response = response.replace("`", "")
    response = response.replace("@", "")
    response = response.replace("#", "")
    response = response.replace("$", "")
    response = response.replace("%", "")
    response = response.replace("^", "")
    response = response.replace("&", "")
    response = response.replace('"', "")
    response = response.replace("'", "")
    return longer_response, response

  @staticmethod
  def export_as_markdown(convo: str, title: str, model: str) -> str:
    """
    Exports a conversation as a Markdown file.

    Args:
      convo (str): The conversation identifier or key.

    Returns:
      str: The path to the Markdown file.

    Notes:
      - The method retrieves the messages from a database 
      based on the given conversation identifier.
      - It iterates through the messages and 
      creates a Markdown file with the following format:
        - The title of the session.
        - The model used for the session.
        - The user's messages.
        - The assistant's messages.
      - The Markdown file is saved in the 'static/markdown/' directory.
    """
    base = g.base
    messages = base.get_conversation_history(convo)
    summary = base.get_conversation_summaries(convo)["short_summary"]
    markdown = ""
    for message in messages:
      if message['role'] == 'system':
        markdown += title + "\n\n"
        markdown += model + "\n\n"
      elif message['role'] == 'user':
        markdown += f"**User:** {message['content']}\n\n"
      elif message['role'] == 'assistant':
        markdown += f"**Assistant:** {message['content']}\n\n"
    filename = f"{summary}.md"
    path = "static/markdown/"
    path_filename = path + filename
    with open(path_filename, "w") as f:
      f.write(markdown)
    return path_filename


class ChatUtils:

  @staticmethod
  def prompt_get(chosen_prompt: str) -> dict:
    """
    Retrieves a prompt and its associated title based on the given prompt key.

    Args:
      prompt (str): The prompt key used to retrieve the corresponding prompt and title.

    Returns:
      dict: A dictionary containing the prompt and title.
            The dictionary has the following structure:
            {
                'prompt': 'The retrieved prompt',
                'title': 'The associated title',
                'opening': 'The opening message'
            }
      If the given prompt key is invalid, a default invalid prompt 
      and title will be returned.
    """
    match chosen_prompt:
      case "prompt4chan":
        prompt = os.environ['4CHANPROMPT']
        title = "4Chan Green Text"
        opening = "Welcome to Green Text. What's your boggle?"
      case "IFSPrompt":
        prompt = os.environ['IFSPROMPT']
        title = "Internal Family Systems AI"
        opening = "Welcome to IFS AI. How are you feeling today?"
      case "KetoPrompt":
        prompt = os.environ['KETOPROMPT']
        title = "Keto Helper"
        opening = "Welcome to the Keto Helper. Do you have any Keto questions?"
      case "CodingBuddy":
        prompt = os.environ['CODINGBUDDYPROMPT']
        title = "Coding Buddy"
        opening = "Hello! I'm your Coding Buddy. What are you working on today?"
      case "TherapistPrompt":
        prompt = os.environ['THERAPISTPROMPT']
        title = "Therapist Bot"
        opening = "Welcome to Therapist Bot. What's on your mind today?"
      case "foodMenuPrompt":
        prompt = os.environ['FOODMENUPROMPT']
        title = "Menu Assistant 8000"
        opening = "Welcome to Menu Assistant 8000. Do you want to create a breakfast, lunch, or dinner menu?" # noqa
      case "HelpfulPrompt":
        prompt = os.environ['HELPFULPROMPT']
        title = "General AI"
        opening = "Hello! I'm a Generally Helpful AI. How can I help you?"
      case "AI_Talks_To_Self":
        prompt = os.environ['TALKTOSELFPROMPT']
        title = "Recursive AI"
      case "CustomPrompt":
        prompt = ""
        title = "Custom Prompt"
      case _:
        prompt = "Invalid Prompt"
        title = "Invalid Title"
        opening = "Invalid Opening"
    return {
      'prompt': prompt,
      'title': title,
      'opening': opening
    }
    
  @staticmethod
  def openai_response(messages: list, model: str = "gpt-3.5-turbo") -> tuple:
    """
    Sends messages to the OpenAI assistant and retrieves the assistant's response.

    Args:
      messages (list): A list of message objects representing the conversation history.
      model (str): The OpenAI model to use for generating the response. 
      (default: "gpt-3.5-turbo")

    Returns:
      tuple: A tuple containing the assistant's response and token usage.
      The tuple has the following structure: (assistant_response, token_usage)

    Raises:
      openai.error.APIError: If the OpenAI API returns an error.
      openai.error.APIConnectionError: If a connection error occurs 
      while communicating with the OpenAI API.
      openai.error.RateLimitError: If the API request exceeds the rate limit.

    Notes:
      - The method uses exponential backoff for retries in case of API errors 
      or connection failures.
      - The method has a maximum retry count and a backoff time between retries.
    """
    retry = True
    retry_count = 0
    max_retries = 5
    backoff_time = 1  # seconds
    assistant_response = ""
    token_usage = 0
    while retry:
      try:
        debug_logger.debug("Attempting to send message to assistant...")
        response = openai.ChatCompletion.create(model=model, messages=messages)
        assistant_response = response["choices"][0]["message"]["content"] # type: ignore
        token_usage = response["usage"]["total_tokens"] # type: ignore
        debug_logger.debug(response["usage"]) # type: ignore
        retry = False
        break
      except openai.error.APIError as e: # type: ignore
        logger.error(f"OpenAI API returned an API Error: {e}")
        retry_count += 1
        if retry_count >= max_retries:
          retry = False
          break
        time.sleep(backoff_time * 2**retry_count)
      except openai.error.APIConnectionError as e: # type: ignore
        logger.error(f"Failed to connect to OpenAI API: {e}")
        retry_count += 1
        if retry_count >= max_retries:
          retry = False
          break
        time.sleep(backoff_time * 2**retry_count)
      except openai.error.RateLimitError as e: # type: ignore
        logger.error(f"OpenAI API request exceeded rate limit: {e}")
        retry_count += 1
        if retry_count >= max_retries:
          retry = False
          break
        time.sleep(backoff_time * 2**retry_count)
    return assistant_response, token_usage

  @staticmethod
  def estimate_tokens(text: str, method: str = "max") -> int | None:
    """
    Estimates the number of tokens required to process the given text.

    Args:
      text (str): The input text to estimate the tokens for.
      method (str): The method to use for estimating the tokens. (default: "max")
                    Supported methods: 'average', 'words', 'chars', 'max', 'min'

    Returns:
        int: The estimated number of tokens required to process the text.

    Notes:
      - The 'max' method returns the maximum of the estimated tokens based 
      on word count and character count.
      - The estimated tokens are rounded up to the nearest integer.
      - An additional 5 tokens are added to the estimated count as a buffer.
    """
    word_count = len(text.split())
    char_count = len(text)
    tokens_count_per_word_est = word_count / 0.6
    tokens_count_char_est = char_count / 4.0
    methods = {
      "average": lambda a, b: (a + b) / 2,
      "words": lambda a, b: a,
      "chars": lambda a, b: b,
      "max": max,
      "min": min
    }
    if method not in methods:
        logger.error("Invalid method.")
        return None
    output = methods[method](tokens_count_per_word_est, tokens_count_char_est)
    return int(output) + 5


class BitcoinUtils:

  @staticmethod
  def get_bitcoin_cost(tokens: int, model: str = "gpt-3.5-turbo") -> int|None:
    """
    Calculates the cost of generating the given number of tokens in Bitcoin.

    Args:
      tokens (int): The number of tokens to calculate the cost for.
      model (str): The OpenAI model used for token generation.(default: "gpt-3.5-turbo")
                    Supported models: "gpt-3.5-turbo", "gpt-4"

    Returns:
      float: The cost in Bitcoin(sats) for generating the specified number of tokens.

    Notes:
      - The cost per 1,000 tokens varies depending on the selected model.
      - The cost is calculated using the current Bitcoin price 
      obtained from the Kraken API.
      - The Kraken API is used to retrieve the BTC to USD exchange rate.
      - The cost is rounded to the nearest whole number of Satoshis (0.00000001 BTC).
    """
    try:
      if model == "gpt-4":
        cost = 0.10  # gpt4 per 1k tokens
      else:
        cost = 0.0099  # chatgpt per 1k tokens
      response, response_json = DataUtils.api_request(
        "GET", 
        "https://api.kraken.com/0/public/Ticker?pair=xbtusd"
      )
      data = response_json["result"]["XXBTZUSD"]["c"]
      price = round(((tokens / 1000) * cost / round(float(data[0]))) / SATS)
      return price
    except Exception as e:
      logger.error(f"Failed to calculate Bitcoin cost: {e}")
      return None

  @staticmethod
  def get_lightning_invoice(sats: int, memo: str) -> dict:
    """
    Generates a Lightning invoice for the given amount of Satoshis.

    Args:
      sats (int): The number of Satoshis to generate the invoice for.
      memo (str): The memo to use for the Lightning invoice.

    Returns:
      dict: The Lightning invoice data.

    Notes:
      - The invoice is generated using the LNBits API.
      - The invoice is generated with a 25 minute expiry.
      - The invoice is generated with a webhook URL to receive payment notifications.
    """
    data = {
      "out": False,
      "amount": sats,
      "memo": memo,
      "expiry": 1500
    }
    try:
      response, response_json = DataUtils.api_request(
        "POST", 
        URL, 
        headers=HEADERS, 
        json=data
      )
      if not response.ok:
        raise Exception("Error:", response.status_code, response.reason)
      return response_json
    except Exception as e:
      logger.error(f"Failed to generate Lightning invoice: {e}")
      return {"Error": "Error generating Lightning invoice."}

  @staticmethod
  def payment_check(payment_hash) -> bool:
    """
    Checks if the given payment hash has been paid.

    Args:
      url (str): The URL of the API endpoint to check the payment status.
      headers (dict): The headers to use for the API request.
      payment_hash (str): The payment hash to check the status for.

    Returns:
      bool: True if the payment has been paid, False otherwise.

    Notes:
      - The API endpoint must return a JSON response with a "paid" key.
      - The API endpoint must return a 200 status code if the payment has been paid.
    """
    try:
      url = f"{URL}{payment_hash}"
      response, response_json = DataUtils.api_request(
        "GET",
        url,
        headers=HEADERS
      )
      if not response.ok:
        raise Exception("Error:", response.status_code, response.reason)
      return response_json.get("paid")
    except Exception as e:
      logger.error(e)
      return False

