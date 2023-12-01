 
from pymilvus import connections, Collection
import cohere  

def prepareDB():
    _HOST = '' #insert server ip here. If it is in local ip is 0.0.0.0
    _PORT = '19530'

    connections.connect(host=_HOST, port=_PORT, db_name="default")
    #print(db.list_database())
    
    collection = Collection("milvus_final")      # Get an existing collection.
    collection.load()
    return collection

def searchDB(query, collection=prepareDB()):
    co = cohere.Client("")  #insert cohere api key
    texts = [query]  
    response = co.embed(texts=texts, model='embed-multilingual-v2.0')  
    embeddings = response.embeddings # All text embeddings 
    
    search_params = {
        "metric_type": "IP", 
        "offset": 0, 
        "ignore_growing": False, 
        "params": {"nprobe": 10}
    }
    
    results = collection.search(
        data=embeddings, 
        anns_field="emb", 
        # the sum of `offset` in `param` and `limit` 
        # should be less than 16384.
        param=search_params,
        limit=4,
        expr=None,
        # set the names of the fields you want to 
        # retrieve from the search result.
        output_fields=['title',"text","url"],
        consistency_level="Strong"
    )
    #print(results)
    data=""
    processed=results[0]
    for a in range(len(processed)):
        data+=(processed[a].entity.get("title")+"\n")
        data+=(processed[a].entity.get("text")+"\n")
        data+=(processed[a].entity.get("url")+"\n")
        data+=(str(processed[a].distance)+"\n\n")

    return data
    
#just to test
# result=searchDB("Abraham Lincoln?")
# print(result)
