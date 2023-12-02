from langchain import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

embeddings = OpenAIEmbeddings()

docsearch = Chroma.from_texts(
    texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))]
)

template = """The following text represents a Database, including Tables, Columns, and their relations such as "Primary Keys" and "Foreign Keys." 
    
Please provide a textual Explain for each table:
    |{Representation}|
    """

promptArchitictureRepresentation = PromptTemplate(
        input_variables=["Representation"],
        template=template,
    )
    
