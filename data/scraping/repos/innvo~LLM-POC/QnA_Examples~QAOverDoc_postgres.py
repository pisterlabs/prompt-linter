from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import psycopg2

# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')

embeddings = OpenAIEmbeddings()
## Set local environment variables
folder_path = "QnA/country_reports/content"
OPENAI_API_KEY=os.getenv("OPEN_API_KEY")
db_user=os.getenv("DBUSER")
db_password=os.getenv("DBPASSWORD")
db_host=os.getenv("DBHOST")

embeddings = OpenAIEmbeddings()

## search emebedding
def search_embedding(query_string):
    print("in search_embedding")
    query_embedding = embeddings.embed_query(query_string)
    return query_embedding

## similarity_search
def similarity_search(query_string):
    print("in similarity_search")
    query_embedding = search_embedding(query_string)
    # print("query_embedding: " + str(query_embedding))
    conn = psycopg2.connect(
        dbname="llm-demo",
        user=db_user,
        password=db_password,
        host=db_host,
        port="5432"
        )
    # Open a cursor to perform database operations
    cur = conn.cursor()
    # Execute aquery
    cur.execute(f" select id, left(embedding_content, 255), 1 - (embedding_vector <=> '{query_embedding}') as cosine_similarity from	public.contentembedding order by 3 desc") 
    # Fetch all results
    results = cur.fetchall()
    # Print results
    for result in results:
        print(result)
    # Commit changes
    conn.commit()
    # Close cursor and connection
    cur.close()
    conn.close()
    return results
  
###############################
query_string = "What did the president say about Ketanji Brown Jackson"
results = similarity_search(query_string)
print(results)

# Quickstart
