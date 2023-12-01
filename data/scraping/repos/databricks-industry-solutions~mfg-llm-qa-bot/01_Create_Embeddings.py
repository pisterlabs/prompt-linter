# Databricks notebook source
# MAGIC %md You may find this notebook on https://github.com/databricks-industry-solutions/mfg-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Create Embeddings
# MAGIC
# MAGIC So that our qabot application can respond to user questions with relevant answers, we will provide our model with content from documents relevant to the question being asked.  The idea is that the bot will leverage the information in these documents as it formulates a response.
# MAGIC
# MAGIC For our application, we've extracted a series of documents from [New Jersey Chemical Data Fact Sheets](https://web.doh.state.nj.us/rtkhsfs/factsheets.aspx). Using this documentation, we have created a vector database that contains an embedded version of the knowledge stored in these sheets.
# MAGIC
# MAGIC <p>
# MAGIC     <img src="https://github.com/databricks-industry-solutions/mfg-llm-qa-bot/raw/main/images/EntireProcess.png" width="700" />
# MAGIC </p>
# MAGIC
# MAGIC
# MAGIC In this notebook, we will load these PDF documents, chunk the entire document into pieces and then create embeddings from this.  We will retrieve those documents along with metadata about them and feed that to a vector store which will create on index enabling fast document search and retrieval.

# COMMAND ----------

# DBTITLE 1,Install required libraries
# MAGIC %pip install -U PyPDF==3.9.1 pycryptodome==3.18.0 langchain==0.0.207 transformers==4.30.1 accelerate==0.20.3  einops==0.6.1 xformers==0.0.20 sentence-transformers==2.2.2 PyCryptodome==3.18.0 typing-inspect==0.8.0 typing_extensions==4.5.0 faiss-cpu==1.7.4 tiktoken==0.4.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load Configs
# MAGIC %run "./utils/configs"

# COMMAND ----------

dbutils.fs.rm(dbfsnormalize(configs['vector_persist_dir']), True)

# COMMAND ----------

# DBTITLE 1,Define helper functions to extract metadata
def extractMetadata(docstr):
  '''
  extracts the common name from the document
  we will use this as metadata for searches
  '''
  dict = {}
  if 'Common Name:' in docstr:
    matches = re.search(r'(?<=Common Name:)(.*?)(?=Synonyms:|Chemical Name:|Date:|CAS Number:|DOT Number:)', docstr)
    if matches is not None and len(matches.groups()) > 0  and matches.groups()[0] is not None :
      dict['Name']=matches.groups()[0].strip()
  return dict


def addMetadataElems(metadict, metadata_i):
  '''add extracted metadata to the metadata collection'''
  if 'Name' in metadict:
    metadata_i['Name']=metadict['Name']
  else:
    metadata_i['Name']=''



# COMMAND ----------

# DBTITLE 1,Create embeddings and store into a vector store (Faiss or ChromaDB)
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import re

#Sentence embeddings. It maps sentences & paragraphs to a 384 dimensional dense vector space
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# vectordb path
vectordb_path = configs['vector_persist_dir']
#source data dir for pdfs
data_dir = configs['data_dir']
# make sure vectordb path is clear
dbutils.fs.rm(dbfsnormalize(vectordb_path), recurse=True)

pathlst = dbutils.fs.ls(dbfsnormalize(data_dir))
display(pathlst)
alldocslstloader=[]
vectordb=None
for idx, path1 in enumerate(pathlst):
  if not str(path1.path).endswith('.pdf'):
    continue
  pdfurl = path1.path.replace(':', '')
  pdfurl = '/' + pdfurl
  #use langchains pypdf loader
  loader = PyPDFLoader(pdfurl)
  alldocslstloader.append(loader)
  #list of documents
  pdfdocs = loader.load()
  cleanpdfdocs = []
  metadict={}
  #clean up doc and also extract metadata.
  for doc in pdfdocs:
    doc.page_content=re.sub(r'\n|\uf084', '', doc.page_content)
    if not metadict: #if already extrcacted then use it.
      metadict = extractMetadata(doc.page_content)
    #recreate with cleaned doc
    cleandoc = Document(page_content = doc.page_content, metadata=doc.metadata)
    #append the doc to list
    cleanpdfdocs.append(cleandoc)
  #It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]
  splitter = RecursiveCharacterTextSplitter(chunk_size=configs['chunk_size'], 
                                            chunk_overlap=configs['chunk_overlap'])
  texts = splitter.split_documents(cleanpdfdocs)
  metadata_lst = []
  ids_lst = []
  pages_lst=[]

  #add metadata to this block
  for idx2, docs in enumerate(texts):
    #add metadata
    metadata_i = {'source': pdfurl, 'source_dbfs' : path1.path}
    #add extracted metadata from doc
    addMetadataElems(metadict, metadata_i)
    metadata_lst.append(metadata_i)
    #add unique id
    ids_i = f'id-{idx2}-{idx+1}'
    ids_lst.append(ids_i)
    pages_lst.append(docs.page_content)
  # define logic for embeddings storage
  # For Chroma
  # vectordb = Chroma.from_texts(
  #   collection_name='mfg_collection',
  #   texts=pages_lst, 
  #   embedding=embeddings, 
  #   metadatas=metadata_lst,
  #   ids=ids_lst,
  #   persist_directory=vectordb_path
  #   )
  # # persist vector db to storage
  # vectordb.persist()
  
  #For FAISS
  if vectordb is None: #first time
    vectordb = FAISS.from_texts(pages_lst, embeddings, metadatas=metadata_lst, ids=ids_lst)
  else: #subsequently add the docs, metadata and ids
    vectordb.add_texts(texts=pages_lst, metadatas=metadata_lst, ids=ids_lst)
vectordb.save_local(vectordb_path)

# COMMAND ----------

# DBTITLE 1,Test the vector store and print all ids stored in db
# Load from Chroma
# vectorstore = Chroma(collection_name='mfg_collection', 
#        persist_directory=vectordb_path,
#        embedding_function=embeddings)


# Load from FAISS
vectorstore = FAISS.load_local(vectordb_path, embeddings)
for key,value in vectorstore.index_to_docstore_id.items():
  print(key, value)


# COMMAND ----------

# DBTITLE 1,Demonstrate similarity search and show sources
def similarity_search(question, filterdict, k=100):
  #fetch_K - Number of Documents to fetch before filtering.
  #filterdict restrict to that subset of docs with that metadata
  matched_docs = vectorstore.similarity_search(question, k=k, filter=filterdict, fetch_k=100)
  sources = []
  content = []
  for doc in matched_docs:
    sources.append(
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
    )
    content.append(doc.page_content)
    

  return matched_docs, sources, content

#test similarity search
matched_docs, sources, content = similarity_search('What happens with acetaldehyde chemical exposure?', {'Name':'ACETALDEHYDE'}, 10)
#print(content)
print(sources)

# COMMAND ----------

#test similarity search
matched_docs, sources, content = similarity_search('what happens if there are hazardous substances?', {'Name':'ACETONITRILE'})
content
sources

# COMMAND ----------

matched_docs
