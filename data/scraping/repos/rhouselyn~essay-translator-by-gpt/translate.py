import os
import threading
import PyPDF2
import openai
import time
import config

API_KEYS = config.API_KEYS
path = config.path
ori_prompt = config.ori_prompt
total_translated_text = []
lock = threading.Lock()
openai.api_key = API_KEYS[0]


# Extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text


# Split the text into chunks
def split_text(text, max_length=1500):
    paragraphs = text.split("\n")
    chunks = []

    i = 0
    while i < len(paragraphs):
        current_chunk = ''
        for paragraph in paragraphs[i:]:
            if len(current_chunk) + len(paragraph) < max_length:
                current_chunk += paragraph + '\n'
                i += 1

        if current_chunk:
            chunks.append(current_chunk)

    return chunks


# Rotate to the next API key
def rotate_api_key():
    current_index = API_KEYS.index(openai.api_key)
    next_index = (current_index + 1) % len(API_KEYS)
    openai.api_key = API_KEYS[next_index]


# Translate a chunk using GPT-3 API with error handling and key rotation
def gpt3_translate(chunk):
    while True:
        try:
            global ori_prompt
            prompt = ori_prompt + '\n\n' + chunk
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            return response
        except openai.error.OpenAIError as e:
            time.sleep(1)  # Wait for a short time before retrying
            print(e)


def gather_translated(chunk, iid):
    global total_translated_text

    rotate_api_key()
    response = gpt3_translate(chunk)
    translated_text = ''

    for res in response:
        translated_text += get_stream(res)

    # Lock is needed for safe access to global variables from multiple threads.
    with lock:
        total_translated_text[iid] = translated_text


def get_stream(chunk):
    if chunk['choices'][0]['finish_reason'] == 'stop':
        return ''
    return chunk['choices'][0]['delta'].get('content')


def main():
    # Step 1: Extract text from PDF
    global path, total_translated_text, API_KEYS
    print('正在解析文件...')
    pdf_text = extract_text_from_pdf(path)
    print('解析完成!')
    name = path.split('\\')[-1].split('.')[0]

    # Step 2: Split the text into chunks
    chunks = split_text(pdf_text)
    total_translated_text = [''] * len(chunks)

    # Step 3 and 4: Translate and save the results
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    file_path = os.path.join(desktop_path, f'translated_{name}.txt')
    length = len(API_KEYS)

    with open(file_path, 'a', encoding='utf-8', errors='ignore') as file:
        print(f'已在桌面创建文档translated_{name}.txt')
        print('开始翻译...')

        is_calculated = False
        i = 0
        while i < len(chunks):
            print(f'\r{i / len(chunks) * 100:.2f}%')

            start_time = time.time()
            threads = []
            for j in range(length):
                if i + j < len(chunks):
                    thread = threading.Thread(target=gather_translated, args=(chunks[i + j], i + j))
                    threads.append(thread)
                    thread.start()

            for thread in threads:
                thread.join()

            for translated_text in total_translated_text[i:i + length]:
                text = translated_text + '\n\n'
                file.write(text)
                file.flush()
                # print(text, end='')

            if not is_calculated:
                is_calculated = True
                interval = time.time() - start_time
                length = max(1, int(interval * len(API_KEYS) // 20))

            i += length

        print('翻译完成!')


if __name__ == "__main__":
    main()
