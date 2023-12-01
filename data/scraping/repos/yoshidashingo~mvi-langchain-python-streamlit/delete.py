from dotenv import load_dotenv
from mvi_langchain import MomentoVectorIndex
from momento import VectorIndexConfigurations, CredentialProvider
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

def delete():
    db = MomentoVectorIndex(embedding_function=OpenAIEmbeddings(),
        configuration=VectorIndexConfigurations.Default.latest(),
        credential_provider=CredentialProvider.from_environment_variable("MOMENTO_AUTH_TOKEN"),
        index_name="sample")
    db._client.delete_index(index_name="sample")
    print('finish delete')
    
delete()