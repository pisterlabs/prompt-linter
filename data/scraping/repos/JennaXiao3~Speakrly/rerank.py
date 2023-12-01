import cohere
co = cohere.Client('lEYa3WjiekUVp1xl02LlSw9oFwosqoXtQE9BVY7J')

def missingPoints(query, docs):
  highs = []
  results = co.rerank(query=query, documents=docs, model='rerank-english-v2.0') 
  
  for idx, r in enumerate(results):
    print(query)
    print(f"Document Rank: {idx + 1}, Document Index: {r.index}")
    print(f"Document: {r.document['text']}")
    print(f"Relevance Score: {r.relevance_score:.2f}")
    print("\n")

    if r.relevance_score > 0:
      highs.append(r.document['text'])
  
  return highs