import argparse
import os
import time
import openai
from dotenv import load_dotenv
load_dotenv()
from utils import *

openai_api_key = os.getenv("open_ai_api_key")

openai.api_key = openai_api_key

def log(x, y = None):
    print(x,y)

def read_text_file(file_path):
    try:
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read the entire content of the file
            file_content = file.read()

            # return the content here
            return file_content

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def write_file(file_path, content):
    try:
        # Open the file in write mode ('w')
        with open(file_path, 'w') as file:
            # Write the content to the file
            file.write(content)

        print(f"File written successfully at: {file_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def main(args):
    start_time = time.time()
    prompt_file_path = str(args.prompt)
    # propmpt_response_path = str(args.response)
    prompt_file = prompt_file_path + '/prompt.txt'
    prompt = read_text_file(prompt_file)
    request_prompt = """""\nWhat forms do I need to file, receive, fill, request, attach, etc, 
    tell me everything I need to do in order to file taxes based on previous tax description?

    In another line just return me the list of forms with corresponding IRS url of each form.
    """
    # response = str(read_text_file(prompt_file)).count
    response = gpt3_completion(prompt + request_prompt)
    write_file(prompt_file_path + '/response.txt', str(response))
    end_time = time.time() - start_time
    print('Time Taken: ', time.time() - start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--prompt', type=str, required=True)
    # parser.add_argument('--response', type=str, required=True)
    args = parser.parse_args()
    main(args)