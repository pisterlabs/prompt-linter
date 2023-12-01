from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from ..content_noc21 import DETAILS_NOC21

nocs=[]
metadatas=[]
for k,v in DETAILS_NOC21.items():
    code="NOC code: \n"+k+"\n\n"
    title="Title: \n"+v["title"]+"\n\n"
    title_examples="Title examples:\n"+"\n".join(v["title_examples"])+"\n\n"
    main_duties="Main duties:\n"+"\n".join(v["main_duties"])
    
    nocs.append(code+title+title_examples+main_duties)
    metadatas.append({"noc_code":k,"title":v["title"],"title_examples":v["title_examples"],"main_duties":v["main_duties"]})
    

embeddings = CohereEmbeddings()


# initialize pinecone
pinecone.init(
    api_key="03754c1d-0a43-4489-946e-d77d90ccf398", 
    environment="us-east4-gcp" 
)

index_name = "noc2021v1"

docsearch = Pinecone.from_texts(nocs, embeddings, metadatas=metadatas, index_name=index_name)


