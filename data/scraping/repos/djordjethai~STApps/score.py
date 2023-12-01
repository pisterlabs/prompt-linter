from tqdm.auto import tqdm
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import os
openai_api_key = os.environ.get('OPENAI_API_KEY')
    # Initialize Pinecone
pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
                  environment=os.environ["PINECONE_API_ENV"])

index_name = 'embedings1'

index = pinecone.GRPCIndex(index_name)


query = "koji su postulati positive doo?"
model=OpenAIEmbeddings()
# create the query vector
xq = model.embed_query(query)

# now query
xc = index.query(xq, top_k=4, include_metadata=True, namespace='positive', index_name='embedings1')
# now query


for result in xc['matches']:
     
     print(f"{round(result['score'], 2)}: {result['metadata']['text']}")