import copy
from transformers import AutoTokenizer, AutoModel
import torch
from text_preprocessing import to_lower, remove_punctuation, lemmatize_word, remove_stopword, lemmatize_word, preprocess_text
from gensim.models.coherencemodel import CoherenceModel

# Define function to compute similarity scores between two embeddings
def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b)

class BERT():
    def __init__(self, docs):
        # Load pre-trained BERT model and tokenizer

        self.docs = docs
        self.model_name = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.top_docs_count = 10
        # Run pre-processing
        self.preprocess_functions = [to_lower, remove_punctuation, lemmatize_word, remove_stopword, lemmatize_word]

        for doc in docs:
            doc["text"] = preprocess_text(doc["text"], self.preprocess_functions)
            doc["text_embedding"] = self.encode_text(doc['text'])



    # Define function to encode text using BERT tokenizer and model
    def encode_text(self, text):
        input_ids = self.tokenizer.encode(text, add_special_tokens=True, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.model(input_ids)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings


    def predict(self, query):
        
        # preprocess
        query = preprocess_text(query, self.preprocess_functions)

        # Encode query text using BERT
        query_embedding = self.encode_text(query)

        temp_dict = copy.deepcopy(self.docs)
        for document in temp_dict:
            similarity_score = cosine_similarity(query_embedding, document["text_embedding"])
            del document['text_embedding']
            document['score'] = similarity_score.item()

        # Sort documents by similarity score in descending order
        temp_dict = sorted(temp_dict, key=lambda x: x['score'], reverse=True)

        return temp_dict[:min(self.top_docs_count, len(temp_dict))]


if __name__ == "__main__":
    # Encode corpus texts using BERT and compute similarity scores
    corpus = [
        {'id': 1, 'text': 'Introduction to machine learning'},
        {'id': 2, 'text': 'Machine learning for beginners'},
        {'id': 3, 'text': 'Deep learning techniques'},
        {'id': 4, 'text': 'Introduction to artificial intelligence'}
    ]

    query_text = "machine learning"
    bert = BERT(docs=corpus)
    result = bert.predict(query_text)
    print(result)