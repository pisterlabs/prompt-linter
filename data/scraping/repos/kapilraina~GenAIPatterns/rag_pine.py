import os, types
import openai
import pinecone


oaimodel = "gpt-3.5-turbo"
oaiembeddingmodel = "text-embedding-ada-002"
OPENAIKEY= os.environ.get('OPENAIKEY')
PINECONEKEY = os.environ.get('PINECONEKEY')
openai.api_key = OPENAIKEY

pcindexname = 'index000003'
openai.api_key = OPENAIKEY
pinecone.init(api_key=PINECONEKEY,environment='northamerica-northeast1-gcp') 
pcindex = pinecone.Index(pcindexname)

def oai_embedding_function(data):
    #print(embeddingsData)
    response = openai.Embedding.create(
    input=data,
    model=oaiembeddingmodel)
    #print(response)
    embeddings = list(map(lambda row: row.embedding,response.data))
    return embeddings

def system_prompt_template(context):
    return (f"Answer the question only using the context provided in 'Context:' section.\n\n"
          + f"Context:\n\n"
          + f"{context}:\n\n")

def collateMatches(m):
    series = ""
    description = ""
    c_ = "title:"+m['title']+";\n author:"+m['author']+";\n rating:"+str(m['rating'])+";\n description:"+ m['description']
    #c_ = "title:"+m['title']+";\n author:"+m['author']+";\n rating:"+str(m['rating'])
    return c_

while True:
    user_input = input("Enter your query: ")
    if user_input == 'Q' or user_input == 'q':
        break
    #queryEmbedding = embedding_function(user_input)
    queryEmbedding = oai_embedding_function(user_input)
    query_response = pcindex.query(
        top_k=5,
        include_values=True,
        include_metadata=True,
        vector=queryEmbedding)
    scores = list((map(lambda m: m.score,query_response.matches)));
    print(scores);
    matchedVectors = list((map(lambda m: str(m['metadata']),query_response.matches)));
    augmentedQuery =  "".join(matchedVectors)    
    
    try:
        completion = openai.ChatCompletion.create(
                    model = oaimodel,
                    messages = [
                                {"role": "system", "content": system_prompt_template(augmentedQuery)},
                                {"role": "user", "content": user_input}
                            ],
                    temperature=0.1
                        )
    except Exception as e:
            print("Error Occured")
            print(e)
    else:
        print("")
        print(completion.choices[0].message.content)
        print("")
        print("[Tokens Consumed "+str(completion.usage.total_tokens)+"]")
        print("")
        indexEliminator = 0