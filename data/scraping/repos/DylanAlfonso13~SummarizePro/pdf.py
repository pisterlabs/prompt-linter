import PyPDF2
import json
import openai


def grab_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    max_characters = 10000
    num_pages = len(pdf_reader.pages)
    extracted_data = []
    total_characters = 0

    for page_num in range(num_pages):
        if total_characters >= max_characters:
            break
        page = pdf_reader.pages[page_num]
        extracted_data.append(page.extract_text())
        total_characters += len(page.extract_text())

    json_data = json.dumps(extracted_data)
    json_output_path = 'path_to_output_json_file.json'

    with open(json_output_path, 'w'):
        json_text = json_data
        return json_text


def divide_pdf(file, chunk_size=3000):
    # Divide the transcript into smaller chunks
    chunks = []
    num_chunks = len(file) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunks.append(file[start:end])

    return chunks


def pdf_summary(text):
    chunks = divide_pdf(text)
    summaries = []
    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Summarize this piece of a text:" +
                    chunk},
            ],
            temperature=1,
        )
        summaries.append(response["choices"][0]["message"]["content"])

    combined_summary = ' '.join(summaries)

    response2 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Connect this collection of summaries "
             + "into one fluid summary" + combined_summary}
        ],
        temperature=1,)
    final_sum = response2["choices"][0]["message"]["content"]
    return final_sum
