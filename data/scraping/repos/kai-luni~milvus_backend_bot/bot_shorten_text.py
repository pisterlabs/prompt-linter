from io import TextIOWrapper
import os
import time
import openai
import json

chatgpt_model_name = os.getenv('CHATGPT_MODEL')
openai.api_type = "azure"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version = os.getenv('OPENAI_API_VERSION')

base_system_message = "You are a helpful assistant."
system_message = f"{base_system_message.strip()}"

def describe_text_in_one_sentence(data):
    """
    This function takes the first 10000 characters from a given data dictionary's text. It then requests GPT to describe 
    the text in one sentence.

    Args:
        data (dict): The data dictionary to process.

    Returns:
        str: The one-sentence description of the text.
    """
    text = data["text"][:10000]  # take the first 10000 characters
    id = data["id"]

    # Define the prompt to ask GPT to describe the text in one sentence
    prompt =f"""
        Beschreibe den folgenden Text in einem Satz:
        ```{text}```
        """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    # Use the send_message function to ask GPT to describe the text in one sentence
    response = send_message(messages, None)
    clean_response = response.replace("ü", "ue").replace("ä", "ae").replace("ö", "oe").replace("ß", "ss")

    return clean_response


def send_message(messages, model_name, max_response_tokens=2500):
    """
    This function sends a message to the OpenAI GPT-3 model and returns the generated response.
    If an exception occurs during the message sending, it waits for 2 seconds and retries the process.
    This function continues retrying until the message is successfully sent.

    Args:
        messages (list): A list of message objects to be sent to GPT-3.
        model_name (str): The name of the GPT-3 model to be used.
        max_response_tokens (int, optional): The maximum length of the generated response. Defaults to 700.

    Returns:
        str: The content of the generated response.
    """

    while True:
        try:
            time.sleep(4)
            response = openai.ChatCompletion.create(
                engine=chatgpt_model_name,
                messages=messages,
                temperature=0.5,
                max_tokens=max_response_tokens,
                request_timeout=30
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"An error occurred: {e}. Waiting for 2 seconds before retrying.")
            time.sleep(2)


def print_conversation(messages):
    for message in messages:
        print(f"[{message['role'].upper()}]")
        print(message['content'])
        print()

def process_and_write_data(data, outfile : TextIOWrapper):
    """
    This helper function processes a given data dictionary. If the text length is more than 2000 characters, 
    it requests GPT to summarize it. It then writes the updated data to a given outfile.

    Args:
        data (dict): The data dictionary to process.
        outfile (file): The file to write the updated data to.

    Returns:
        None
    """
    id = data["id"]
    text = data["text"]
    print(f"Process: {id}")
    
    print(f">>> length text: {len(text)}")
    prompt =f"""
        Aender den Text so als wenn ein Berndt ihn geschrieben hat. Berndt schreibt Texte die mit wenig Worten alles wissenswerte uebermitteln. Schreibe ihn auf Deutsch. Lasse keine Informationen aus.
        Text: ```{text}```
        """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    response = send_message(messages, None)
    clean_response = response.replace("ü", "ue").replace("ä", "ae").replace("ö", "oe").replace("ß", "ss")
    data["text"] = clean_response
    print(f">>> length response: {len(clean_response)}")
    
    # write updated data to the JSONL file immediately
    json.dump(data, outfile)
    outfile.write('\n')
    outfile.flush()

def split_and_process_data(data, outfile : TextIOWrapper, one_sentence):
    """
    This helper function splits a given data dictionary's text into equal pieces. The number of pieces is determined by
    rounding up the total text length divided by 2500. It then processes each piece and writes the updated data to 
    the given outfile.

    Args:
        data (dict): The data dictionary to process.
        outfile (file): The file to write the updated data to.
        one_sentence (string): a summary of the data

    Returns:
        None
    """
    text = data["text"]
    id = data["id"]

    # Determine the number of chunks needed
    num_chunks = len(text) // 2500
    if len(text) % 2500 != 0:
        num_chunks += 1

    # Determine the chunk size so that all chunks are of equal size
    chunk_size = len(text) // num_chunks
    if len(text) % num_chunks != 0:
        chunk_size += 1

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        chunk_text = text[start_index:end_index]
        chunk_id = f"{id}_{i+1}"
        chunk_data = {"id": chunk_id, "text": f"{one_sentence}. {chunk_text}"}

        process_and_write_data(chunk_data, outfile)

def summarize_large_texts():
    """
    This function reads a JSONL file line by line, and for each line, if the text length is more than 2000 characters, 
    it requests GPT to summarize the text to less than 2000 characters. If the text length is more than 2000 characters, 
    it keeps splitting the text into two halves and processing each half until the text length is less than 2000 characters. 
    The function writes the updated data to a new JSONL file immediately after processing each line.

    Note: This function assumes the existence of a function named 'send_message' which communicates with GPT and 
    a variable named 'chatgpt_model_name' which specifies the GPT model to be used.

    Args:
        None

    Returns:
        None
    """

    with open('gpt/phat_user.jsonl', 'r') as jsonl_file:
        with open('gpt/updated_file.jsonl', 'a') as outfile:
            process = False
            for line in jsonl_file:
                data = json.loads(line)  # convert JSON string to Python dictionary
                #manually jump back to certain line in json file by setting the key here
                id = data["id"]
                if id == f"Soeren_Stein":
                    process = True
                if not process:
                    continue

                if len(data["text"]) < 150:
                    continue
                print(f"write summary for {id}")
                one_sentence = describe_text_in_one_sentence(data)
                #print(f"the summary is: {one_sentence}")
                split_and_process_data(data, outfile, one_sentence)




    
if __name__ == "__main__":
    summarize_large_texts()
