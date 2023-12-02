from datasets import load_dataset
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
import os
import langchain.schema as lc_schema


if 'chroma_db' not in os.listdir('.'):

	large_context_df = load_dataset('odunola/foodie-large-context')
	small_context_df = load_dataset('odunola/foodie-small-context')



	texts_1 = large_context_df['train']['texts']
	texts_2 = small_context_df['train']['texts']


	#build embedding function, adjust bath_size to optimise or accommodate GPU VRAM
	embedding_function = HuggingFaceBgeEmbeddings(model_name='thenlper/gte-large', encode_kwargs = {"batch_size":64, "show_progress_bar":True})


	large_documents = [lc_schema.Document(page_content = i) for i in texts_1]
	small_documents = [lc_schema.Document(page_content = i) for i in texts_2]


	#build chroma database and make persistted in storage
	index_1 = Chroma.from_documents(large_documents, embedding_function, persist_directory="./chroma_db", collection_name = "foodie")
	index_2 = Chroma.from_documents(small_documents, embedding_function, persist_directory="./chroma_db", collection_name = 'foodie_small')


