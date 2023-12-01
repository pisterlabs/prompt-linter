import os

import torch
import openai
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2Model

model_amlv = SentenceTransformer('all-MiniLM-L6-v2')
model_ambv = SentenceTransformer('all-mpnet-base-v2')
model_gpt2 = GPT2Model.from_pretrained('gpt2')

openai_api_key = os.getenv("OPENAI_API_KEY")
embedding = None
if openai_api_key is not None:
    openai.api_key = openai_api_key
    embedding = openai.Embedding("text-davinci-002")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def generate_embedding_amlv(text: str) -> torch.Tensor:
    """
    Generates an embedding for a given text using the all-MiniLM-L6-v2 model from SentenceTransformers.

    Args:
    - text (str): A string representing text to generate an embedding for.

    Returns:
    - embedding (torch.Tensor): A tensor representing the embedding for the input text.
    """

    # Generate embedding
    embedding = model_amlv.encode(text, show_progress_bar=True)

    return embedding


def generate_embedding_ambv(text: str) -> torch.Tensor:
    """
    Generates an embedding for a sentence using the all-mpnet-base-v2 model from SentenceTransformers.

    Args:
    - sentence (str): A string representing the sentence to generate an embedding for.

    Returns:
    - embedding (torch.Tensor): A torch Tensor representing the embedding for the input sentence.
    """

    # Generate embedding
    embedding = model_ambv.encode(text, show_progress_bar=True)

    return embedding


def generate_embedding_gpt2(text: str) -> torch.Tensor:
    """
    Returns the embeddings of the input text using the pre-trained GPT-2 model.

    Args:
    text (str): The input text to convert to embeddings.

    Returns:
    torch.Tensor: A tensor of shape (1, 768) containing the embeddings of the input text.
    """
    # Tokenize input text and convert to tensor
    inputs = tokenizer(text, return_tensors='pt')

    # Get model outputs and extract the last hidden state
    outputs = model_gpt2(**inputs)
    last_hidden_state = outputs.last_hidden_state

    # Average the last hidden state over the sequence dimension to get a single embedding vector
    embeddings = torch.mean(last_hidden_state, dim=1)

    return embeddings


def generate_embedding_openai(text: str) -> torch.Tensor:
    """
    Performs text embedding using the OpenAI API.

    Args:
    - text (str): The input text to perform embedding on.

    Returns:
    - torch.Tensor: The resulting text embedding as a tensor.
    """
    assert embedding is not None
    response = embedding.embed_text(text)
    return torch.tensor(response)
