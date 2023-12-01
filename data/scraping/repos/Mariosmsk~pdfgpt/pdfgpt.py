"""Main module."""
import os
import openai
import tiktoken
import textwrap
import pandas as pd
from pypdf import PdfReader
from openai.embeddings_utils import get_embedding, cosine_similarity


class PDFBot:

    def __init__(self, openai_key):
        self.filename = 'embedding data'
        openai.api_key = openai_key
        self.slice_stop = 20000

    # Extract text from PDFs
    def generateText(self, file_path=None, df=None):
        if df is not None:
            extracted_text, num_pages = self.extract_any_text(df)
            return extracted_text, num_pages

        self.filename = os.path.basename(file_path)

        pdf_document = PdfReader(file_path)
        num_pages = len(pdf_document.pages)
        extracted_text = ''
        for page_num in range(num_pages):
            processed_text = []
            current_font_size = None
            current_blob_text = ''
            page = pdf_document.pages[page_num]
            page_text = page.extract_text()
            page_text = [
                {'font_size': None, 'text': page_text, 'x_coord': None,
                 'y_coord': None}] if page_text is not None else []

            for text in page_text:
                if text['font_size'] == current_font_size:
                    current_blob_text += f" {text['text']}"
                    if len(current_blob_text) >= 2000:
                        processed_text.append({
                            'font_size': current_font_size,
                            'text': current_blob_text,
                            'page': page_num
                        })
                        current_font_size = None
                        current_blob_text = ''
                else:
                    if current_font_size is not None and len(current_blob_text) >= 1:
                        processed_text.append({
                            'font_size': current_font_size,
                            'text': current_blob_text,
                            'page': page_num
                        })
                    current_font_size = text['font_size']
                    current_blob_text = current_blob_text
            extracted_text += text['text']
        text_chunks = self.generateChunkText(extracted_text)
        return text_chunks, num_pages

    def split_text_to_pages(self, text):
        extracted_text = textwrap.wrap(text, width=self.slice_stop)
        return extracted_text

    def extract_any_text(self, df):
        string_columns = list(df.keys())
        text_column = df[string_columns].apply(lambda x: ' '.join([str(x[col]) for col in x.index]), axis=1)
        extracted_text = ' '.join(text_column.tolist())
        extracted_text = self.split_text_to_pages(extracted_text)
        # extracted_text = [{'text': page_text} for page_text in extracted_text if page_text.strip() != '']
        num_pages = len(extracted_text)
        return extracted_text, num_pages

    def generateChunkText(self, extracted_text):
        # Initialise tokenizer
        tokenizer = tiktoken.get_encoding("cl100k_base")
        chunks = self.create_chunks(extracted_text, 1000, tokenizer)
        text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
        return text_chunks

    ###################################################
    # Source: Openai - https://github.com/openai/openai-cookbook
    # Split a text into smaller chunks of size n, preferably ending at the end of a sentence
    def create_chunks(self, text, n, tokenizer):
        tokens = tokenizer.encode(text)
        """Yield successive n-sized chunks from text."""
        i = 0
        while i < len(tokens):
            # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
            j = min(i + int(1.5 * n), len(tokens))
            while j > i + int(0.5 * n):
                # Decode the tokens and check for full stop or newline
                chunk = tokenizer.decode(tokens[i:j])
                if chunk.endswith(".") or chunk.endswith("\n"):
                    break
                j -= 1
            # If no end of sentence found, use n tokens as the chunk size
            if j == i + int(0.5 * n):
                j = min(i + n, len(tokens))
            yield tokens[i:j]
            i = j
    ###################################################

    # Generate embeddings from PDFs
    def generateEmbeddings(self, extracted_text='', model_embeddings="text-embedding-ada-002"):
        df = pd.DataFrame(extracted_text)
        df[0] = df[0].str.slice(stop=self.slice_stop )
        df['text_length'] = df[0].str.len()
        embeddings = df[0].apply(lambda x: get_embedding(x, engine=model_embeddings))
        df["embeddings"] = embeddings
        return df

    # Generate prompt
    def generatePrompt(self, df, num_pages, message, model_embeddings="text-embedding-ada-002"):
        query_embedding = get_embedding(message, engine=model_embeddings)
        similarities = cosine_similarity(df.embeddings.tolist(), query_embedding)
        result_indices = similarities.argsort()[::-1][:num_pages]
        # df[0].array[0] = df[0].array[0][:4000]
        result = df.iloc[result_indices]
        min_len = min(result['text_length'][0:3])
        if min_len > 4000:
            min_len = 4000
        try:
            if len(result) == 1:
                res1 = result.iloc[0][0][:7000]
            else:
                res1 = result.iloc[0][0][:4000]

        except:
            res1 = result.iloc[0][0][:min_len]

        try:
            try:
                if min_len < 2000:
                    res2 = result.iloc[1][0][:4000 - min_len]
                else:
                    res2 = result.iloc[1][0][:3500]
            except:
                res2 = result.iloc[1][0][:min_len]
        except:
            res2 = ''
        try:
            try:
                if min_len < 1000:
                    res3 = result.iloc[2][0][:1000 - min_len]
                else:
                    res3 = result.iloc[2][0][:500]
            except:
                res3 = result.iloc[2][0][:min_len]
        except:
            res3 = ''
        prompt = f"""Given the question: {message} and the following embeddings as data:
                           1. {res1}
                           2. {res2}
                           3. {res3}
                       Give an answer based only on the data where I provide or return \"Not specified\".
                       """
        return prompt

    # Sends a prompt to the OpenAI API and return the response
    def sendPrompt(self, prompt, model="gpt-3.5-turbo", temperature=0.9, max_tokens=1500):

        # response = openai.Completion.create(
        #     model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens,
        #     top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)

        response = openai.ChatCompletion.create(
            messages=[
                {'role': 'system', 'content': f'You answer questions about the {self.filename}.'},
                {'role': 'user', 'content': prompt},
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response['choices'][0]['message']['content']
