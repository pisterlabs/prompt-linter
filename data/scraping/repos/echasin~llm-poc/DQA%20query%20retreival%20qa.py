# importing libraries
import os
from dotenv import load_dotenv
import openai
import pinecone
import langchain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

# Load the environment variables from the .env file.
load_dotenv()

# Get the value of the API_KEY environment variable.

# OpenAI credentials
api_key = os.getenv('openai_api_key')
# Pinecone credentials

api_key = os.getenv('api_key')
environment = os.getenv('environment')
index_name = os.getenv('index_name')

# Create the vector store object
pinecone.init(
    api_key= api_key, # type: ignore
    environment= environment # type: ignore
)

index_name = "techplaybook-dev"
print("index_name: ", index_name)


# Cnnect to the Pinecone Index
index = pinecone.Index(index_name)
# Create the embeddings object
# Create the embeddings object
embeddings = OpenAIEmbeddings(model ="text-embedding-ada-002") # type: ignore

#Finding Similar Documents using Pinecone Index
docsearch = Pinecone.from_existing_index(index_name, embeddings)

model_name = "text-davinci-003"
# model_name = "gpt-3.5-turbo"
llm = OpenAI(model=model_name) # type: ignore
#llm = openai.ChatCompletion.create(model=model_name)
chain = load_qa_chain(llm, chain_type="stuff")

# query = "Who is database expert and what are they known for?"
# query = "Who is Ralph Kibmball and what are they known for?"
# query = "Who is Ralph Kibmball and what are they known for?  Do not compare to other people."  
# query = "What is a data warehouse? Answer in bullets.  Provide explaination for each bullet."  
# query = "What is a data warehouse? Answer in bullets.  Provide explaination for each bullet. Write answers like you are talking to very technical engineer"
query = "What is a Bills house? Answer in bullets. If you don't have any context and are unsure of the answer, reply that you don't know about this topic and are always learning."
# query = "how many books did Bill Innmon write a book on Data Lakehous? If yes what was the name of the book? Whats was the publish date?"

def get_answer(query):
    similar_docs = docsearch.similarity_search(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

def get_similiar_docs(query, k=2, score=True):
    if score:
        similar_docs = docsearch.similarity_search_with_score(query, k=k)
    else:
        similar_docs = docsearch.similarity_search_with_score(query, k=k)
    return similar_docs


docs = docsearch.similarity_search(query)
 
# print(docs[0].page_content)

# found_docs = docsearch.max_marginal_relevance_search(query, k=2, fetch_k=10)
# for i, doc in enumerate(found_docs):
#     print(f"{i + 1}.", doc.page_content, "\n")

print("###########################")
answer = get_answer(query)
print("query: ", query)
print(answer)
print("###########################")
#################


res = openai.Embedding.create(
    input=[query],
    engine="text-embedding-ada-002"
)

xq = res['data'][0]['embedding'] # type: ignore

# get relevant contexts (including the questions)
res = index.query(xq, top_k=5, include_metadata=True)
     
print("XXXXXXXXXXXXXXXXXXXXXXXXXX")
print(res)

# get list of retrieved text
contexts = [item['metadata']['text'] for item in res['matches']]

augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query

print(augmented_query)
print("XXXXXXXXXXXXXXXXXXXXXXXXXX")
print("-------------------------")
# system message to 'prime' the model
primer = f"""You are Q&A bot. A highly intelligent system that answers
user questions based on the information provided by the user above
each question. If the information can not be found in the information
provided by the user you truthfully say "I don't know".
"""

res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)


print(res['choices'][0]['message']['content']) # type: ignore
print("-------------------------")