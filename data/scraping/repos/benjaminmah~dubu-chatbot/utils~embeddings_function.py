from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
# from openai.embeddings_utils import get_embedding
import openai
from tqdm import tqdm

class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="text-embedding-ada-002"):
        self.model_name = model_name

    def get_embedding(self,text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    
    def __call__(self, texts: Documents) -> Embeddings:
        texts = [t.replace("\n", " ") for t in texts]

        tracked_texts = tqdm(texts)
        tracked_texts.set_description('Obtaining embeddings')

        embeddings = [self.get_embedding(t, self.model_name) for t in tracked_texts]
        return embeddings