from dotenv import load_dotenv
load_dotenv()

from langchain.document_loaders import TextLoader
loader = TextLoader('OneFlower/花语大全.txt')

from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

query = "玫瑰花的花语是什么？"
result = index.query(query)
print(result)