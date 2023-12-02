import openai, os

from llama_index import SimpleDirectoryReader, VectorStoreIndex

openai.api_key = os.environ.get("OPENAI_API_KEY")

documents = SimpleDirectoryReader('./data/mr_fujino').load_data()
index = VectorStoreIndex.from_documents(documents)

index.storage_context.persist('index_mr_fujino.json')


response = index.query("鲁迅先生去哪里学的医学？")
print(response)