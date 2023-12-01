import os
import time
from dotenv import load_dotenv
import openai
from tqdm import tqdm
from openAI import process_text
import re

# Load environment variables from .env file
load_dotenv(".env")

def parse_prompt_file(prompt_file):
    with open(prompt_file, 'r') as file:
        lines = file.readlines()
        i = 0
        prompt = ""
        while i < len(lines) and not lines[i].startswith("Temperature:"):
            prompt += lines[i]
            i += 1
        temperature = float(lines[i].split(":")[1].strip()) if i < len(lines) else float(os.getenv('DEFAULT_TEMP'))
    return (prompt, temperature)

def main():
    # Get environment variables
    openai_api_key = os.getenv('OPENAI_API_KEY')
    output_directory = os.getenv('OUTPUT_DIRECTORY')
    output_separator = os.getenv('OUTPUT_SEPARATOR')
    prompt_directory = os.path.abspath(os.getenv('PROMPT_DIR'))
    audio_directory = os.path.abspath(os.getenv('AUDIO_DIRECTORY'))

    # Set the OpenAI API key
    openai.api_key = openai_api_key

    # Get all audio files in the specified directory
    audio_files = [audio_file_name for audio_file_name in os.listdir(audio_directory) if not audio_file_name.endswith('.DS_Store')]

    # Process all audio files in the specified directory with progress bar
    def process_function(file_name):
        return process_audio_file(file_name, prompt_directory, output_directory, output_separator, audio_directory)

    process_files(audio_files, process_function)

def process_audio_file(audio_file_name, prompt_directory, output_directory, output_separator, audio_directory):
    input_filepath = os.path.join(audio_directory, audio_file_name)
    output_filepath = os.path.join(output_directory, f"{os.path.splitext(audio_file_name)[0]}.txt")

    # Transcribe audio file
    with open(input_filepath, 'rb') as audio_file:
        print(f"Processing file: {input_filepath}")
        response = openai.Audio.transcribe('whisper-1', audio_file)

    # Get prompt number from filename
    prompt_number = int(os.path.splitext(audio_file_name)[0].split('-')[0])

    # Get prompt and temperature for prompt number
    prompt_file = None
    for file_name in os.listdir(prompt_directory):
        match = re.match(r"(\d+)", file_name)  # match one or more digits
        if match:
            number_part = int(match.group(1))
            if number_part == prompt_number:
                prompt_file = os.path.join(prompt_directory, file_name)
                break

    if prompt_file is None:
        print(f"No prompt found for number '{prompt_number}'")
        return

    prompt, temperature = parse_prompt_file(prompt_file)

    # Process text with OpenAI
    output_text = process_text(prompt, response, temperature)

    # Write output to file
    with open(output_filepath, 'w') as file:
        file.write(f"{response}\\n{output_separator}\\n{output_text}")

def process_files(file_list, process_function):
    total_files = len(file_list)
    successful_files = 0

    with tqdm(total=total_files, desc="Processing files", ncols=100) as pbar:
        for file_name in file_list:
            try:
                process_function(file_name)
                successful_files += 1
                tqdm.write(f"'{file_name}' successfully processed.")
            except Exception as e:
                tqdm.write(f"Error processing '{file_name}': {str(e)}")
            pbar.update(1)
            time.sleep(0.1)

    print(f"{successful_files} of {total_files} files successfully processed.")

if __name__ == "__main__":
    main()