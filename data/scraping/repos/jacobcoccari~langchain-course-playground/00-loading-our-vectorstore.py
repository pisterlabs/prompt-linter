from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings

url_list = [
    "https://www.youtube.com/watch?v=8Nn5uqE3C9w",
    "https://www.youtube.com/watch?v=szxPar0BcMo",
    "https://www.youtube.com/watch?v=jvnU0v6hcUo",
    "https://www.youtube.com/watch?v=UUCEeC4f6ts",
    "https://www.youtube.com/watch?v=0LsrkWDCvxg",
]

documents = []

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=50,
)

for url in url_list:
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=True,
    )
    loader = loader.load_and_split(text_splitter=text_splitter)
    documents = documents + loader

embedding_function = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-base",
)

db = Chroma.from_documents(
    documents,
    embedding_function,
    persist_directory="./10-Retriver-Augmented-Generation/crash-course-db",
)

db.persist()

docs = db.similarity_search("who was ashoka?", k=2)
print(docs)
