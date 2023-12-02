from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

loader = TextLoader('data/ai.txt')

index = VectorstoreIndexCreator().from_loaders([loader])


query = "When was AI founded as an academic discipline?"

result = index.query(query)


print(result)