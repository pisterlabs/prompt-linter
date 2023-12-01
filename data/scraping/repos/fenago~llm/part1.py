from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings()
text = '''
Meet Dr. Ernesto Lee
Dr. Lee is an impassioned data scientist, futurist and technologist and the original founding member of Learning Voyage. Dr. Lee’s career illustrates a lifelong commitment to pushing the envelope on innovation and growing opportunities for all those around him. With a passion for technology and teaching, Dr. Lee has also written books and courses on Blockchain, Programming, Big Data Science and more. Dr. Lee’s work as a public speaker, writer and author has quickly seen him emerge to be one of America’s most exciting thought leaders and expert voices in the field of Data Science and technology.
'''

doc_embeddings = embeddings.embed_documents([text])

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# print(OPENAI_API_KEY)
print(doc_embeddings)