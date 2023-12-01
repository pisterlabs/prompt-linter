import years_query as yq
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


llm = OpenAI(openai_api_key="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")

def y2022_search(query):
    y2022_cons = yq.y2022_filter(query)
    yq.data_cleaner(y2022_cons)

    #txt_loader
    input_file = 'NTU_library.csv'  # 輸入的CSV檔案名稱
    output_file = 'NTU_library.txt'  # 輸出的TXT檔案名稱
    yq.merge_page_content(input_file, output_file)
    loader = TextLoader('NTU_library.txt')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    #print(documents[0])

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)

    y2022_search = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    return y2022_search