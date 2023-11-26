class OpenAIEmbeddings:
    def __init__(self, client, openai_api_key=None, model="text-embedding-ada-002"):
        from openai import OpenAI
        
        self.client = client
        self.model = model
        
    def embed_query(self, text):
        text = [text.replace("\n", " ")]
        return self._get_embedding(text, model=self.model)[0].embedding
        
    def embed_documents(self, documents):
         embeddings = self._get_embedding(documents, model=self.model)
         return [emb.embedding for emb in embeddings]
    
    def _get_embedding(self, text, model):
        return self.client.embeddings.create(input = text, model=model).data
