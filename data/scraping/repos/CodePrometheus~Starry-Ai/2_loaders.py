from langchain.llms import OpenAI

llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0,
    max_tokens=1024,
    verbose=True,
)
# chat = ChatOpenAI(
#     temperature=0,
#     verbose=True
# )

"""
CSV Loader
"""

from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='./example_data/mlb_teams_2012.csv')
# Specify a column to be used identify the document source
loader = CSVLoader(file_path='./example_data/mlb_teams_2012.csv', source_column="Team")
data = loader.load()

"""
Directory Loader 加载目录中的所有文档
"""
from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader('../', glob="**/*.md")
docs = loader.load()
len(docs)

# By default this uses the UnstructuredLoader class. However, you can change up the type of loader pretty easily.
from langchain.document_loaders import TextLoader

loader = DirectoryLoader('../', glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()
len(docs)

"""
HTML 文档
"""
from langchain.document_loaders import UnstructuredHTMLLoader

loader = UnstructuredHTMLLoader("example_data/fake-content.html")
data = loader.load()

from langchain.document_loaders import BSHTMLLoader

loader = BSHTMLLoader("example_data/fake-content.html")
data = loader.load()

"""
PDF
"""
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("example_data/layout-parser-paper.pdf")
pages = loader.load_and_split()
pages[0]

# Fetching remote PDFs using Unstructured
from langchain.document_loaders import OnlinePDFLoader

loader = OnlinePDFLoader("https://arxiv.org/pdf/2302.03803.pdf")
data = loader.load()
print(data)

"""
YouTube%20Loader.ipynb
"""

from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain

# pip install youtube-transcript-api
# pip install pytube

loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=QsYGlZkevEg", add_video_info=True)
result = loader.load()
print(type(result))
print(f"Found video from {result[0].metadata['author']} that is {result[0].metadata['length']} seconds long")
print("")
print(result)

chain = load_summarize_chain(llm, chain_type="stuff", verbose=False)
chain.run(result)

# Long Videos
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(result)
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
chain.run(texts[:4])
