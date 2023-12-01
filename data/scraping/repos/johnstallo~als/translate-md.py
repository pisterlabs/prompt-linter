import os
import time
import threading
import argparse
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
MAX_TOKEN_LENGTH = os.environ.get("MAX_TOKEN_LENGTH")
if MAX_TOKEN_LENGTH is None:
    MAX_TOKEN_LENGTH = 256 # default
else:
    MAX_TOKEN_LENGTH = int(MAX_TOKEN_LENGTH)

def chunk_text(text):
    paragraphs = text.split('\n')
    chunks = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            # If the paragraph is empty (i.e., an empty line), add it as a separate chunk
            chunks.append('')
            continue

        words = paragraph.split()
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) > MAX_TOKEN_LENGTH:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1  # +1 for space

        # Add the last chunk of the paragraph
        if current_chunk:
            chunks.append(' '.join(current_chunk))

    # return chunks

    # Generate blocks
    current_length = 0;
    current_block = ""
    blocks = []
    for chunk in chunks:
        # print(f"current_length: {current_length}, chunk: {chunk}, current_block: {current_block}")
        if current_length + len(chunk) < MAX_TOKEN_LENGTH:
            current_block += chunk + '\n'
            current_length = len(current_block)
        else:
            blocks.append(current_block)
            current_block = '\n' + chunk + '\n'
            current_length = len(chunk)
            # current_block = ""
            # current_length = 0

    if current_block:
        blocks.append(current_block)

    return blocks


def call_api_with_timeout(chunk, language, timeout_duration):
    result = {"success": False, "response": None, "error": None}
    MODE = os.environ.get("MODE")
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL")
    if OPENAI_MODEL is None:
        OPENAI_MODEL = "gpt-3.5-turbo"

    def api_call():
        try:
            if MODE != "Mock":
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a translator model that translates Christian bible studies. Keep the same markdown where possible. If you come across an image tag, then keep it as-is. Don't translate scripture quotes, copy the scripture passage verbatim from the best available bible version in the target language."},
                        {"role": "user", "content": f"Translate it to {language}: {chunk}"}
                    ],
                    temperature=0,
                    max_tokens=MAX_TOKEN_LENGTH
                )
                result["response"] = response.choices[0].message.content
                result["success"] = True
            else:
                result["response"] = chunk
                result["success"] = True
        except Exception as e:
            result["error"] = str(e)

    for _ in range(3):  # Retry up to 3 times
        thread = threading.Thread(target=api_call)
        thread.start()
        thread.join(timeout_duration)
        if thread.is_alive():
            print(f"Timeout reached for chunk. Attempting retry...")
            thread.join()  # Optionally, you may choose to kill the thread instead
        else:
            break  # Break out of the loop if successful or an error occurs

        if result["success"] or result["error"]:
            break  # Break out of the loop if successful or an error occurs

    return result

def translate_text_to_file(text_chunks, language, output_file_path):
    total_chunks = len(text_chunks)
    timeout_duration = 120  # 2 minutes in seconds
    MODE = os.environ.get("MODE")

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for i, chunk in enumerate(text_chunks, 1):
            if not chunk.strip():
                if i == 1 or text_chunks[i - 2].strip():
                    output_file.write('\n')
                continue

            print(f"Translating chunk {i} of {total_chunks} ({len(chunk)} tokens)...")
            # print(f"{chunk}")

            # DEBUG!
            # if i > 3:
            #     break


            response = call_api_with_timeout(chunk, language, timeout_duration)
            if response["success"]:
                output_file.write(response["response"] + '\n')
            else:
                if response["error"]:
                    output_file.write(f"An error occurred: {response['error']}\n")
                else:
                    output_file.write("Failed to translate due to timeout.\n")

            if i < total_chunks:
                pause_time = (MAX_TOKEN_LENGTH / 90000 * 60)
                if pause_time > 1 and MODE != "Mock":
                    print(f"    Pausing translation for {pause_time:.1f} seconds for ChatGPT rate limiting")
                    time.sleep(pause_time)



def main():
    start_time = time.time() # Capture the start time

    # parser = argparse.ArgumentParser(description="Translate text from a Markdown file.")
    # parser.add_argument("file_path", help="Path to the Markdown file for translation")

    # args = parser.parse_args()
    # file_path = args.file_path

    file_path = os.environ.get("MD_FROM_DOCX_ORIGINAL")
    if file_path is None:
        print(f"The markdown file to translate has not been specified. Set MD_FROM_DOCX_ORIGINAL in the .env file.")
        return
    
    output_file_path = os.environ.get("TRANSLATED_MD")
    if output_file_path is None:
        print(f"The translated markdown filepath has not been specified. Set TRANSLATED_MD in the .env file.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_to_translate = file.read()
    except FileNotFoundError:
        print(f"File not found at '{file_path}'. Please check the file path.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    language = os.environ.get("TARGET_LANGUAGE")
    if language is None:
        language = 'Italian' # default
    
    # Report what we're doing
    print(f"Translating {file_path} to {language} at chunks of {MAX_TOKEN_LENGTH} tokens.")

    # Chunk up source
    text_chunks = chunk_text(text_to_translate)

    # Translate chunks and write to a file
    translate_text_to_file(text_chunks, language, output_file_path)

    end_time = time.time() # Capture the end time
    total_time = (end_time - start_time) / 60
    print(f"Translation completed in {total_time:.2f} minutes.")

if __name__ == "__main__":
    main()
