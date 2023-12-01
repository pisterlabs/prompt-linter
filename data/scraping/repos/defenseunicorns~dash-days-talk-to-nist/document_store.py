import os

# Chroma
import chromadb
import openai
from chromadb import Settings
from transformers import BartTokenizer, BartForConditionalGeneration

import ingest


class DocumentStore:
    def __init__(self):
        self.index_name = "TalkToNist"
        openai.api_key = 'Free the models'
        openai.api_base = os.environ.get('REMOTE_API_URL')
        self.client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory="db"))
        self.load_required = not self.client.get_collection(name="TalkToNist")
        self.collection = self.client.get_or_create_collection(name="TalkToNist")
        self.ingestor = ingest.Ingest(self.index_name, self.client, self.collection)
        self.summary_model_name = 'facebook/bart-large-cnn'
        self.summary_tokenizer = BartTokenizer.from_pretrained(self.summary_model_name)
        self.summary_model = BartForConditionalGeneration.from_pretrained(self.summary_model_name)
        # For the sliding window
        self.chunk_size = 200
        self.overlap_size = 50

    def query(self, query_text):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=3
        )

        docs = [t for t in results['documents'][0]]
        combined_document = ' '.join(docs)

        # Split the combined document into overlapping chunks
        chunks = self.chunk_text(combined_document, self.chunk_size, self.overlap_size)
        summaries = [self.summarize(chunk) for chunk in chunks]
        flat_summaries = self.flat_map_summaries(chunks, summaries)
        combined_summary = ' '.join(flat_summaries)

        return combined_summary

    def flat_map_summaries(self, chunks, summaries):
        flat_summaries = []
        for chunk, summary in zip(chunks, summaries):
            flat_summaries.extend(summary)
        return flat_summaries

    def load_pdf(self, path):
        if self.load_required:
            ingest.load_data(self.ingestor, path)

    # Function for summarization
    def summarize(self, document, size_multiplier=7):
        inputs = self.summary_tokenizer(document, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = self.summary_model.generate(inputs['input_ids'], num_beams=4, min_length=30, max_length=size_multiplier * len(document.split()),
                                     early_stopping=True)
        return [self.summary_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

    def chunk_text(self, text, chunk_size, overlap_size):
        tokens = text.split(' ')
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap_size):
            chunk = ' '.join(tokens[i:i + chunk_size])
            chunks.append(chunk)
        return chunks