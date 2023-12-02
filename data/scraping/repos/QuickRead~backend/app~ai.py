import openai
from transformers import AutoTokenizer
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

openai.api_key = "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"


class TextSummarizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    # NOTE: from https://blog.devgenius.io/how-to-get-around-openai-gpt-3-token-limits-b11583691b32
    def break_up_file_to_chunks(self, text, chunk_size=1500, overlap=100):
        
        tokens = self.tokenizer.encode(text)
        num_tokens = len(tokens)
        
        chunks = []
        for i in range(0, num_tokens, chunk_size - overlap):
            chunk = tokens[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks, num_tokens

    def summarize(self, text: str):
        prompt_response = []

        chunks, num_tokens = self.break_up_file_to_chunks(text)
        print(num_tokens)

        for i, chunk in enumerate(chunks):
            prompt_request = "Sumarizuj tuto část článku ČESKY: " + self.tokenizer.decode(chunk)
            messages = [{"role": "system", "content": "Budeš sumarizovat text."}]    
            messages.append({"role": "user", "content": prompt_request})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.5,
                max_tokens=500,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0
            )
            prompt_response.append(response["choices"][0]["message"]['content'].strip())


        prompt_request = "Udělej sumarizaci následujících informací ČESKY: " + str(prompt_response)
        messages = [{"role": "system", "content": "Budeš sumarizovat text."}]
        messages.append({"role": "user", "content": prompt_request})

        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=.5,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        summary = response["choices"][0]["message"]['content'].strip()

        return summary


if __name__ == "__main__":
    CONTENT = None
    if not CONTENT:
        with open('./app/doc.txt', 'r') as f:
            CONTENT = f.read()

    summarizer = TextSummarizer()
    summary = summarizer.summarize(CONTENT)
    print(summary)
