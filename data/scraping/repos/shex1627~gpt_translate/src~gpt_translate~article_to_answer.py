import concurrent.futures
import openai  # Make sure you've imported the required module
from functools import partial

def send_single_chatcomplete_api(message_list, chatcomplete_config):
    try:
        response = openai.ChatCompletion.create(**chatcomplete_config, messages=message_list)
        return response
    except Exception as e:
        print(f"Error sending chatcomplete API request: {e}")

def parallel_send_chatcomplete_api(message_chunk, chatcomplete_config, num_threads=4):
    """
    get answer for multiple questions in parallel
    """
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        func = partial(send_single_chatcomplete_api, chatcomplete_config=chatcomplete_config)
        for response in executor.map(func, message_chunk):
            if response:
                responses.append(response)
    return responses
