from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# Vector Store Index to create our database about our knowledge
# LLamaCpp embeddings from the Alpaca model
from langchain.embeddings import LlamaCppEmbeddings
# FAISS  library for similaarity search
from langchain.vectorstores.faiss import FAISS
import os  #for interaaction with the files
import datetime
import pandas as pd


# assign the path for the 2 models GPT4All and Alpaca for the embeddings 
gpt4all_path = './models/gpt4all-converted.bin' 
llama_path = './models/ggml-model-q4_0.bin' 
# Calback manager for handling the calls with  the model
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# create the embedding object
embeddings = LlamaCppEmbeddings(model_path=llama_path)
# create the GPT4All llm object
llm = GPT4All(model=gpt4all_path, callback_manager=callback_manager, verbose=True)


# Original code was for PDFs, here: http://webcache.googleusercontent.com/search?q=cache:nfL4yHE0LoQJ:https://artificialcorner.com/gpt4all-is-the-local-chatgpt-for-your-documents-and-it-is-free-df1016bc335&client=safari&hl=en&gl=us&strip=1&vwsrc=0
# Manually prompted ChatGPT-4 code for lists of strings
def split_chunks(sources):
    """
    Args:
        sources: list of strings
    """
    chunk_size = 256
    overlap_size = 32

    chunks = []
    for source in sources:
        # Split the string source into chunks
        for i in range(0, len(source), chunk_size-overlap_size):
            chunk = source[i:i+chunk_size]
            chunks.append(chunk)

    return chunks


# def similarity_search(query, index):
#     # k is the number of similarity searched that matches the query
#     # default is 4
#     matched_docs = index.similarity_search(query, k=3) 
#     sources = []
#     for doc in matched_docs:
#         sources.append(
#             {
#                 "page_content": doc.page_content,
#                 "metadata": doc.metadata,
#             }
#         )

#     return matched_docs, sources



# ChatGPT output to modify original code from PDF to CSV

# get the list of csv files from the docs directory into a list format
csv_folder_path = './data'
doc_list = [s for s in os.listdir(csv_folder_path) if s.endswith('.csv')]
num_of_docs = len(doc_list)
general_start = datetime.datetime.now() #not used now but useful
print("starting the loop...")
loop_start = datetime.datetime.now()
print("generating first vector database and then iterate with .merge_from")

# Load CSV and read the specific column
# df = pd.read_csv(os.path.join(csv_folder_path, doc_list[0]))

df = pd.read_csv("data/df_sellout_clean.csv", sep="\t")

# For failures log CSV
# docs = df['Description'].tolist()  # Replace 'column_name' with the name of your column

docs = df.iloc[:,0].to_list()
chunks = split_chunks(docs)
db0 = FAISS.from_texts(chunks, embeddings)


# I don't want multiple CSVs at the moment
# print("Main Vector database created. Start iteration and merging...")
# for i in range(1, num_of_docs):
#     print(doc_list[i])
#     print(f"loop position {i}")

#     # Load CSV and read the specific column
#     df = pd.read_csv(os.path.join(csv_folder_path, doc_list[i]))
#     docs = df['column_name'].tolist()  # Replace 'column_name' with the name of your column
#     chunks = split_chunks(docs)
#     dbi = FAISS.from_texts(chunks, embeddings)
    
#     print("start merging with db0...")
#     db0.merge_from(dbi)



loop_end = datetime.datetime.now() #not used now but useful
loop_elapsed = loop_end - loop_start #not used now but useful
print(f"All documents processed in {loop_elapsed}")
print(f"the database is done with {num_of_docs} subset of db index")
print("-----------------------------------")
print(f"Merging completed")
print("-----------------------------------")
print("Saving Merged Database Locally")
# Save the databasae locally
db0.save_local("index_company_names")
print("-----------------------------------")
print("merged database saved")
general_end = datetime.datetime.now() #not used now but useful
general_elapsed = general_end - general_start #not used now but useful
print(f"All indexing completed in {general_elapsed}")
print("-----------------------------------")