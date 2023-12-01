import re
import json
from typing import List, Callable
import logging

import openai
from tqdm import tqdm

logger = logging.getLogger("article translation util")

def text_split(text: str, max_length = 1024) -> List[str]:
    """
    Split a large trunck of text into smaller trunk based on max_length. 
    """

    # Split the text into sentences using a regular expression pattern
    sentences = re.findall('[^。！？]+[。！？]?', text)

    # Initialize a list to hold the segments
    segments = []

    # Initialize a variable to hold the current segment
    current_segment = ''

    # Loop through each sentence in the text
    for sentence in sentences:
        # If the current segment plus the current sentence is less than the maximum length,
        # add the current sentence to the current segment
        if len(current_segment) + len(sentence) <= max_length:
            current_segment += sentence
        # Otherwise, add the current segment to the list of segments and start a new segment with the current sentence
        else:
            segments.append(current_segment)
            current_segment = sentence

    # Add the last segment to the list of segments
    segments.append(current_segment)

    # Return the list of segments
    return segments

def combine_openapi_response(response):
    """
    Combines all the response messages in a list from an OpenAPI response.

    Args:
        response (requests.Response): The OpenAPI response to parse.

    Returns:
        A list of all the response messages in the OpenAPI response.
    """
    # Try to get the response JSON using response.to_dict_recursive()
    try:
        response_json = response.to_dict_recursive()
    except:
        # If response.to_dict_recursive() fails, try to parse the response content as JSON
        try:
            response_json = response.json()
        except ValueError:
            # If parsing the response content as JSON fails, return the response content as a string
            return [response.content.decode()]

    # Check for errors in the response
    if "error" in response_json:
        return [response_json["error"]]

    # Check for success in the response
    if "choices" in response_json:
        messages = []
        for choice in response_json["choices"]:
            if "message" in choice:
                message = choice["message"]
                if "content" in message:
                    messages.append(message["content"])
        if messages:
            return messages

    # Handle unknown response format
    return [response.content.decode()]

def send_chatcomplete_api(message_chunk, chatcomplete_config):
    """
    Sends OpenAI chatcomplete API for each message in a message chunk using the specified config.

    Args:
        message_chunk (list): A list of lists of messages to send to the OpenAI chatcomplete API.
        chatcomplete_config (dict): A dictionary of necessary parameters for the OpenAI chatcomplete API.

    Returns:
        A list of all the responses from the OpenAI chatcomplete API.
    """

    # Send OpenAI chatcomplete API for each message in message_chunk
    responses = []
    with tqdm(total=len(message_chunk), desc="Sending chatcomplete API requests") as pbar:
        for message_list in message_chunk:
            try:
                response = openai.ChatCompletion.create(**chatcomplete_config, messages=message_list)
                responses.append(response)
            except Exception as e:
                print(f"Error sending chatcomplete API request: {e}")
            pbar.update(1)

    return responses


def translate_article(article_text: str, completion_config: dict, messages_generator: Callable[[str], List[str]]) -> str:
    """
    Translates an article text by breaking it up into smaller trunks and sending each trunk
    to the OpenAI chatcomplete API using the specified completion config and message generator.

    Args:
        article_text (str): The article text to translate.
        completion_config (dict): A dictionary of necessary parameters for the OpenAI chatcomplete API.
        messages_generator (callable): A function that takes a text string as input and generates a list of
            messages to send to the OpenAI chatcomplete API.

    Returns:
        The translated article text as a single string.
    """
    print("split test into trunks")
    text_trunks = text_split(article_text)
    print(f"there are {len(text_trunks)} text_trunks")
    trunk_messages = [messages_generator(text_trunk) for text_trunk in text_trunks]
    responses = send_chatcomplete_api(trunk_messages, completion_config)
    trunk_translations = [combine_openapi_response(response)[0] for response in responses]
    article_translation = "\n".join(trunk_translations)
    return article_translation


def process_article(article_text: str, text_chunk_size: int, completion_config: dict, messages_generator: Callable[[str], List[str]]) -> str:
    """
    Process an article text by breaking it up into smaller trunks and sending each trunk
    to the OpenAI chatcomplete API using the specified completion config and prompt(message_generator).

    Args:
        article_text (str): The article text to process.
        completion_config (dict): A dictionary of necessary parameters for the OpenAI chatcomplete API.
        messages_generator (callable): A function that takes a text string as input and generates a list of
            messages to send to the OpenAI chatcomplete API.

    Returns:
        The processed text as a single string.
    """
    text_trunks = text_split(article_text, max_length=text_chunk_size)
    trunk_messages = [messages_generator(text_trunk) for text_trunk in text_trunks]
    responses = send_chatcomplete_api(trunk_messages, completion_config)
    trunk_processed = [combine_openapi_response(response)[0] for response in responses]
    article_processed = "\n".join(trunk_processed)
    return article_processed 
