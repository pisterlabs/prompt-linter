def get_split_documents(docs, chunk_size, chunk_overlap):
  from langchain.text_splitter import RecursiveCharacterTextSplitter

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)

  return text_splitter.split_documents(docs)