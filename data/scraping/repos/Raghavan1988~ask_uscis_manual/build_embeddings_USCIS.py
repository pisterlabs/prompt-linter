import os
from langchain.embeddings import OpenAIEmbeddings
import langchain
from annoy import AnnoyIndex
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer, util
import sys
from langchain.document_loaders import PyPDFLoader


embeddings = OpenAIEmbeddings(openai_api_key="ENTER OPEN AI KEY")
model = SentenceTransformer('sentence-transformers/allenai-specter', device='cpu')

def get_embeddings_per_page(page):
    try:
        ret = embeddings.embed_query(page)
        return ret
    except:
        return None

policy_manual = sys.argv[1]
loader = PyPDFLoader(policy_manual)
pages = loader.load_and_split()

print("There are " + str(len(pages)) + " pages in the policy manual.")


embeddings_dict = {}
embeddings_dict2 = {}
for i in range(0,len(pages)):
    e = get_embeddings_per_page(pages[i].page_content)
    embeddings_dict[i] = e
    embeddings_dict2[i] = model.encode(pages[i].page_content)
    if (i % 100 == 0):
        print("Finished embedding pages " + str(i))


t = AnnoyIndex(1536, 'angular')
t2 = AnnoyIndex(768, 'angular')
index_map = {}
i = 0
for i in embeddings_dict:
    t.add_item(i, embeddings_dict[i])
    t2.add_item(i, embeddings_dict2[i])
    index_map[i] = pages[i]

t.build(2000)
name1= "Policy Manual" + "_ada.ann"
t.save(name1)
t2.build(len(pages))
name2 = "Policy Manual" + "_specter.ann"
t2.save(name2)


