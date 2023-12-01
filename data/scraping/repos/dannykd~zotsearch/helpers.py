import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from dotenv import load_dotenv
import pinecone

def searchFromPinecone(query, n):
  load_dotenv()
  try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
    index = pinecone.Index("zotsearch")
    res = index.query(
      vector=get_embedding(
          query,
          engine="text-embedding-ada-002"),
      top_k=n,
      include_metadata=True
    )
  except:
      raise Exception("Something went wrong with our API calls")
  
  try:
    courses = []
    for match in res["matches"]:
      courseMatch = {"id": match["id"],
                      "title": match["metadata"]["title"],
                      "desc": match["metadata"]["description"],
                      "dept": match["metadata"]["department"]}
      courses.append(courseMatch)

  except:
     raise Exception("Something went wrong when retrieving the courses")
      
  return courses
  

if __name__ == "__main__":
    print(searchFromPinecone("Front end development or database management or networking stuff", 20))
