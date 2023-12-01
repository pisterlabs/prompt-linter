from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.pgvector import PGVector,DistanceStrategy
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import os
import psycopg
from CustomLoaders import MyReadTheDocsLoader, FreshdeskLoader
import sqlalchemy
from sqlalchemy.orm import Session

def initdb():
	# initialize db
	embedding = OpenAIEmbeddings()
	dbname=os.environ.get("POSTGRES_DATABASE", "postgres")
	host=os.environ.get("POSTGRES_HOST", "127.0.0.1")
	port=os.environ.get("POSTGRES_PORT", "5432")
	user=os.environ.get("POSTGRES_USER", "postgres")
	password=os.environ.get("POSTGRES_PASSWORD", "postgres")

	connection_string = 'dbname={dbname} host={host} port=5432 user={user} password={password}'
	connection_string = connection_string.format(
        dbname=dbname,
        host=host,
        port=port,
        user=user,
        password=password
    )

	connection = psycopg.connect(connection_string)
	connection.autocommit = True
	cursor = connection.cursor()
	cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
	
	CONNECTION_STRING = PGVector.connection_string_from_db_params(
    	driver="psycopg2",
    	host=host,
    	port=port,
    	database=dbname,
    	user=user,
    	password=password,
	)
	vectordb = PGVector(
		connection_string=CONNECTION_STRING,
		embedding_function=embedding,
		distance_strategy=DistanceStrategy.COSINE
	)
	vectordb.create_vector_extension()
	vectordb.create_tables_if_not_exists()
	vectordb.create_collection()
	return vectordb

def data_ingest(vectordb: VectorStore):
	# check if there are new sources in sources folder
	sources = os.listdir("sources")

	for source in sources:
		# skip source if it is already in db
		try:
			with Session(vectordb._conn) as session:
				statement = sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector")
				session.execute(statement)
				session.commit()
			cursor=vectordb._conn.cursor()
			cursor.execute("select count(*) from public.langchain_pg_embedding where cmetadata->>'source' LIKE 'sources/"+source+"%';")
			if cursor.fetchone()[0] > 0:
				continue
		except:
			pass
		print("Ingesting new source: ", source)
		# ingest new sources
		if source.endswith(".txt"):
			# load txt data
			loader = TextLoader("sources/"+source)
			pass
		elif source.endswith(".it") or source.endswith(".org"):
			# Load ReadTheDocs data
			loader = MyReadTheDocsLoader("sources/"+source, features='html.parser', encoding='utf-8', errors='ignore')
		elif source.endswith(".freshdesk.json"):
			loader = FreshdeskLoader("sources/"+source)
		else:
			continue
		
		documents = loader.load()
		# get all sources already in db
		already_ingested = []
		try:
			cursor.execute("select distinct(cmetadata->>'source') from public.langchain_pg_embedding;")
			already_ingested = cursor.fetchall()
		except:
			pass
		# filter out already ingested sources
		documents = [doc for doc in documents if doc.metadata["source"] not in already_ingested]
		# Add file permissions to metadata
		for doc in documents:
			doc.metadata["owner"] = os.stat(doc.metadata["source"]).st_uid
			doc.metadata["group"] = os.stat(doc.metadata["source"]).st_gid
			doc.metadata["permissions"] = oct(os.stat(doc.metadata["source"]).st_mode)[-3:]
			
		text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=100)
		texts = text_splitter.split_documents(documents)
		vectordb.add_documents(texts)

if __name__ == "__main__":
	# initialize vector db
	vectordb = initdb()
	# ingest new data
	data_ingest(vectordb)

	# chat with the user
	from langchain.chains.llm import LLMChain
	from langchain.callbacks.manager import CallbackManager
	from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
	from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
	from langchain.chains.question_answering import load_qa_chain

	# Construct a ConversationalRetrievalChain with a streaming llm for combine docs
	# and a separate, non-streaming llm for question generation
	llm = OpenAI(temperature=0)
	streaming_llm = OpenAI(streaming=True,
			callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
			verbose=True,
			temperature=0
			)
	question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
	doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=QA_PROMPT)

	qa = ConversationalRetrievalChain(
    	retriever=vectordb.as_retriever(),
		combine_docs_chain=doc_chain,
		question_generator=question_generator,
		return_source_documents=True
		)

	chat_history = []
	query = input("Hi! I'm your personal data manager. Ask me something about your data.\n")
	result = qa({"question": query, "chat_history": chat_history})
	print("\n")

	while True:
		sources = []
		for doc in result["source_documents"]:
			sources.append(doc.metadata["source"])
			#print("Source: "+doc.metadata["source"])
			#print("Owner: "+str(doc.metadata["owner"]))
			#print("Grup: "+str(doc.metadata["group"]))
			#print("Permissions: "+str(doc.metadata["permissions"]))
		sources = list(set(sources))
		[print("Source: "+source) for source in sources]
		
		print("-"*80)
		
		chat_history = [(query, result["answer"])]
		query = input("\n")
		result = qa({"question": query, "chat_history": chat_history})
		print("\n")
	
