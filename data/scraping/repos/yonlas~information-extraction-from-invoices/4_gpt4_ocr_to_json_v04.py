import os
import json
import openai
import nltk
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)

# OpenAI API key
openai.api_key = os.getenv('OPENAI_KEY')

# Configuration for OpenAI parameters
CONFIG = {
    "temperature": 0,
    "max_tokens_output": 800,
    "model_name": "gpt-4",
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop_sequences": ["}"],
    "n": 1,
    "stream": False
}

instruction = """I want you to act as a data annotator. 
    Your task is to put the extracted text from the energy invoice into the specified json format. 
    You should only change the "input" and output the desired json format.
    The "charge_period_end_date" should always be later than the "charge_period_start_date". 
    If you cannot find one of the keys in the extracted text, simply do not output the whole line for it. (e.g. if you cannot find the "mpan" key, please do not output "mpan": "",).
    Never output "any key": "not found" or "any key": "null" if you did not find it in the extracted text.
    The desired format:
    {
        "vendor_name": "input", # only the vendor/supplier name, do not input the "Vendor Name" label
        "invoice_date": "input", # input the date of the invoice in the format dd/mm/yyyy (e.g. 01/01/2021). Note that the extracted text may have a different format, but you should enter it as dd/mm/yyyy.
        "invoice_number": "input", # only the invoice number, do not input the "Invoice Number" label
        "total_amount": "input", # should be a number with 2 decimal places (e.g. 12,345.67), do not input the currency symbol.
        "charge_period_start_date": "input", # input the date of the invoice in the format dd/mm/yyyy (e.g. 01/01/2021). Note that the extracted text may have a different format, but you should enter it as dd/mm/yyyy.
        "charge_period_end_date": "input", # input the date of the invoice in the format dd/mm/yyyy (e.g. 01/01/2021). Note that the extracted text may have a different format, but you should enter it as dd/mm/yyyy.
        "mpan": "input", # do not input more than 13 characters which are usually only digits (e.g. "1234567890123"), it may contain spaces or dashes and be broken down into groups of 2, 4, 4 and 3 characters.
        "account_number": "input" # only the account number, do not input the "Account Number" label
    }
    The extracted text:""" 


def count_tokens_in_text(text):
    """Count the number of tokens in a text using nltk."""
    tokens = nltk.word_tokenize(text)
    return len(tokens)

def count_tokens_in_file(file_path):
    """Count the number of tokens in a file using nltk."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return count_tokens_in_text(text)

def process_text_with_gpt4(text):
    instruction_token_count = count_tokens_in_text(instruction)
    max_tokens_input = 8000 - instruction_token_count - CONFIG["max_tokens_output"]

    text_token_count = count_tokens_in_text(text)
    # Logging token counts
    logging.info(f"Instruction token count: {instruction_token_count}")
    logging.info(f"Text token count: {text_token_count}")

    try:
        response = openai.ChatCompletion.create(
            model=CONFIG["model_name"],
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": text}
            ],
            max_tokens=CONFIG["max_tokens_output"],
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
            frequency_penalty=CONFIG["frequency_penalty"],
            presence_penalty=CONFIG["presence_penalty"],
            stop=CONFIG["stop_sequences"],
            n=CONFIG["n"],
            stream=CONFIG["stream"]
        )
        return response.choices[0].message['content']
    except Exception as e:
        logging.error(f"Error occurred while processing with OpenAI: {str(e)}")
        return None


def write_to_json(file_path, file_name, data):
    output_data = {
        "file_name": file_name,
        "gpt_4_predictions": json.loads(data + "}")
    }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)


def call_openai_with_retry(text, max_retries=5, base_wait_time=3):
    """Call the OpenAI API with a retry mechanism."""
    for retry in range(max_retries):
        try:
            return process_text_with_gpt4(text)
        except openai.error.OpenAIError as e:
            if "rate_limit_exceeded" in str(e):
                # Calculate wait time using exponential backoff
                wait_time = base_wait_time * (2 ** retry)
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # If it's another type of error, raise it
                raise
    logging.error(f"Failed after {max_retries} retries.")
    return None

def truncate_text_to_fit(text, max_tokens):
    """Truncate the text to fit within the specified token limit."""
    tokens = nltk.word_tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return ' '.join(truncated_tokens)

def main():
    input_folder = './ocrs'
    output_folder = './gpt_4_outputs'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    error_files = []

    for filename in os.listdir(input_folder):
        # time.sleep(5)
        if filename.endswith('.txt'):
            # Check if the file was already processed
            output_file_path = os.path.join(output_folder, filename.replace('.txt', '.json'))
            if os.path.exists(output_file_path):
                logging.info(f"File {filename} was already processed. Skipping...")
                continue

            txt_path = os.path.join(input_folder, filename)
            logging.info(f"Processing file: {filename}")
            
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()

                file_token_count = count_tokens_in_file(txt_path)
                logging.info(f"Tokens in the file from ocrs folder: {file_token_count}")

                instruction_token_count = count_tokens_in_text(instruction)
                max_tokens_input = 6500 - instruction_token_count - CONFIG["max_tokens_output"]
                
                if file_token_count + instruction_token_count > max_tokens_input:
                    logging.warning(f"File {filename} exceeds the token limit. Truncating...")
                    allowable_tokens = max_tokens_input - instruction_token_count
                    extracted_text = truncate_text_to_fit(extracted_text, allowable_tokens)

                processed_text = call_openai_with_retry(extracted_text)
                if processed_text:
                    write_to_json(output_file_path, filename, processed_text)
                    logging.info(f"This file was processed: {filename}")

            except Exception as e:
                logging.error(f"Error occurred while processing {filename}: {str(e)}")
                error_files.append(filename)

    if not error_files:
        print("All files were processed successfully!")
    else:
        print(f"Errors occurred while processing {len(error_files)} files.")
        for err_file in error_files:
            print(f"- {err_file}")

if __name__ == '__main__':
    main()

