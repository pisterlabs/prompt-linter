import os
import time
from pypdf import PdfReader
import re
import openai
from nltk import sent_tokenize
from tqdm import tqdm
import tiktoken


tokenizer = tiktoken.get_encoding("cl100k_base")

# Replace this with your OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    number_of_pages = len(reader.pages)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def preprocess_text(text):
    # Remove noise
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'^\s*|\s*$', '', text)

    # Split the text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)

    # Process each paragraph
    processed_text = ''
    for paragraph in paragraphs:
        # Remove unnecessary spaces and line breaks
        paragraph = re.sub(r'\s+', ' ', paragraph).strip()

        # Remove anything resembling a website URL
        paragraph = re.sub(r'http[s]?://\S+|www\.\S+', '', paragraph)

        # Append the cleaned paragraph to the processed text
        processed_text += paragraph + '\n'

    # Return the processed text
    return processed_text


def nearest_sentence_end(chunk, remaining_text):
    remaining_sentences = sent_tokenize(remaining_text, language="english")
    completed_chunk = chunk

    for sentence in remaining_sentences:
        if len(tokenizer.encode(completed_chunk + sentence)) <= 1500:
            completed_chunk += ' ' + sentence
        else:
            break

    return completed_chunk


def chunk_text(text, max_tokens=1500):
    tokens = tokenizer.encode(text)
    chunks = []
    current_chunk_tokens = []
    current_tokens = 0

    remaining_text = text

    for token in tokens:
        if current_tokens + 1 <= max_tokens:
            current_chunk_tokens.append(token)
            current_tokens += 1
            token_text = tokenizer.decode([token])
            remaining_text = remaining_text[len(token_text):]
        else:
            chunk = tokenizer.decode(current_chunk_tokens).strip()
            chunk = nearest_sentence_end(chunk, remaining_text)
            chunks.append(chunk)
            current_chunk_tokens = [token]
            current_tokens = 1
            remaining_text = remaining_text.lstrip()

    if current_chunk_tokens:
        chunk = tokenizer.decode(current_chunk_tokens).strip()
        chunk = nearest_sentence_end(chunk, remaining_text)
        chunks.append(chunk)

    return chunks


def chat_with_gpt4(prompt, conversation_history):
    time.sleep(1.5)  # Add a 1.5-second delay between requests to avoid overloading the rate limits
    response = openai.ChatCompletion.create(
        model='gpt-4-32k',
        messages=conversation_history + [{"role": "user", "content": f"Summarize the section of the research paper:\n{prompt}"}],
        temperature=.8,
    )
    return response["choices"][0]["message"]["content"]


def summarize_pdf(pdf_folder_path):
    summary_text = ''

    file_iterator = (f for f in os.listdir(pdf_folder_path) if f.endswith(".pdf"))
    for filename in file_iterator:
        pdf_path = os.path.join(pdf_folder_path, filename)
        pdf_text = extract_text_from_pdf(pdf_path)
        preprocessed_text = preprocess_text(pdf_text)
        text_chunks = chunk_text(preprocessed_text)

        conversation_history = [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": "I will be sending you iterative sections of a single research paper, for each section I want you to provide detailed, bulletproof summaries that capture the entire essence of what is being discussed within that specific section."},
            {"role": "assistant", "content": "Got it, I will summarize each section into a concise, accurate representation of its content. Send the sections whenever you're ready!"},
        ]

        complete_summary = ""

        with tqdm(total=len(text_chunks), desc="Summarizing PDF") as pbar:
            summary_message = {"role": "assistant", "content": ""}
            conversation_history.append(summary_message)

            for chunk_index, chunk in enumerate(text_chunks):
                response_text = chat_with_gpt4(chunk, conversation_history)

                complete_summary += f"\n\nChunk {chunk_index + 1} Summary:\n{response_text}"

                summary_message["content"] += f"\n\nChunk {chunk_index + 1} Summary:\n{response_text}"
                pbar.update(1)

        summary_text += f"\nComplete summary for {os.path.basename(pdf_path)}:\n{complete_summary}\n"

    return summary_text


def main():
    pdf_folder_path = input("Enter the path to the folder containing the PDFs to summarize: ")
    summary_text = summarize_pdf(pdf_folder_path)
    print(summary_text)


if __name__ == "__main__":
    main()