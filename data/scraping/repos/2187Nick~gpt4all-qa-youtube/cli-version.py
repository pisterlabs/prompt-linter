from langchain.llms import GPT4All
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma 
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# enter the path to your model here
select_model = 'models/gpt4all-converted.bin' # ./models/gpt4all-converted.bin'

# enter your youtube link here
youtube_link = "https://www.youtube.com/watch?v=GhRNIuTA2Z0" 

# set the length of the response you want
results_length = 30
# increase this number to get more random results
model_temp = 0.3

loader = YoutubeLoader.from_youtube_channel(youtube_link, add_video_info=True)

documents = loader.load()
title = documents[0].metadata['title']
print(documents[0].metadata['title'])

# chromadb can give error with publish date. So deleting it
del documents[0].metadata['publish_date']

# Split the documents into chunks of 100 characters
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

documents = text_splitter.split_documents(documents)

# remove special characters
k = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_";
title = title.replace(" ", "_") 
getVals = list(filter(lambda x: x in k, title))
title = "".join(getVals)
 
# set the directory to store the vectorized database
persist_directory = title

# select an embeddings transformer  # list of models and their speeds : https://www.sbert.net/docs/pretrained_models.html
#model_name = "sentence-transformers/all-mpnet-base-v2"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# if the vectorized database doesn't exist then create it. Else load it
if not os.path.exists(persist_directory):
    db_vector = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
else:
    db_vector = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

llm = GPT4All(model=select_model, n_predict=results_length, temp=model_temp) # n_ctx=512, n_threads=8, n_predict=100, temp=.8

retriever = db_vector.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever) #, chain_type_kwargs=chain_type_kwargs)

parse_answer = lambda x: x["result"].split("Answer:")[1].strip()

while True:
    query = input("Query: ")

    # Query the youtube video data 
    result = qa(query, return_only_outputs=True)
    print("Youtube Sourced Answer: ", parse_answer(result))
