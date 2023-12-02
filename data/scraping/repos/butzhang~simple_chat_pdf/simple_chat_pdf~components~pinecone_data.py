# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.chains import ChatVectorDBChain
# from langchain.document_loaders import PyPDFLoader
# from langchain.vectorstores import Pinecone
# import pinecone
# from simple_chat_pdf.constant import PINECONE_API_KEY, PINECONE_ENVIRONMENT, OPENAI_API_KEY, PINE_CONE_INDEX_NAME
# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# loader = PyPDFLoader('./case_studies.pdf')
# documents = loader.load_and_split(text_splitter)
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# vector_pinecone = Pinecone.from_documents(documents, embeddings,
#                                           index_name=PINE_CONE_INDEX_NAME,
#                                           namespace="pdf-second-version"
#                                           )