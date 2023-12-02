from langchain.vectorstores import Pinecone
import openai
import pinecone
import tqdm

class PineconeUpsert:
    """Wraps around main functionality of upsert text to pinecone index"""
    def __init__(self, PINECONE_API_KEY, PINECONE_ENVIRONMENT,INDEX_NAME):

        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        if INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(INDEX_NAME, dimension=1536)

        self.index = pinecone.Index(INDEX_NAME)
        self.embedding_model = "text-embedding-ada-002"

    def upsert(self, text:str):
        """Upserts text to pinecone index"""
        self.create_and_index_embeddings(text, self.embedding_model, self.index)

    def create_and_index_embeddings(self,data:str, model, index):
        print(f'Inserting: {data[:30]}...')
        batch_size = 32  # process everything in batches of 32
        for i in tqdm(range(0, len(data), batch_size)):
            # set end position of batch
            i_end = min(i+batch_size, len(data))
            # get batch of lines and IDs
            lines_batch = data[i: i+batch_size]
            ids_batch = [str(n) for n in range(i, i_end)]
            # create embeddings
            res = openai.Embedding.create(input=lines_batch, engine=model)
            embeds = [record['embedding'] for record in res['data']]
            # prep metadata and upsert batch
            meta = [{'text': line} for line in lines_batch]
            to_upsert = zip(ids_batch, embeds, meta)
            # upsert to Pinecone
            index.upsert(vectors=list(to_upsert))
