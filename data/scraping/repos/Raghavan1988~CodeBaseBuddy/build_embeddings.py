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

embeddings = OpenAIEmbeddings(openai_api_key="")
model = SentenceTransformer('sentence-transformers/allenai-specter', device='cpu')


##name = "langchain"
##GITHUB_PATH = "/home/raghavan/langchain"

##name = "open_interpreter"
##GITHUB_PATH = "/home/raghavan/open-interpreter"

name = sys.argv[1]
GITHUB_PATH = sys.argv[2]

def get_files(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if ".py" in file or ".sh" in file or ".java" in file:
                files.append(os.path.join(r, file))
    return files


def get_file_embeddings(path):
    try:
        text = get_file_contents(path)
        ret = embeddings.embed_query(text)
        return ret
    except:
        return None

def get_file_contents(path):
    with open(path, 'r') as f:
        return f.read()


print (name)
print (GITHUB_PATH)
files = get_files(GITHUB_PATH)
print(len(files))

embeddings_dict = {}
embeddings_dict2 = {}
i = 0
s = set()
for file in files:
    e = get_file_embeddings(file)
    if (e is None):
        print ("Error in embedding file: ")
        print (file)
        s.add(file)
    else:
        embeddings_dict[file] = e
        embeddings_dict2[file] = model.encode(get_file_contents(file))
    i+=1
    if (i%100 == 0):
        print ("No of files processed: " + str(i))


t = AnnoyIndex(1536, 'angular')
t2 = AnnoyIndex(768, 'angular')
index_map = {}
i = 0
for file in embeddings_dict:
    t.add_item(i, embeddings_dict[file])
    t2.add_item(i, embeddings_dict2[file])
    index_map[i] = file
    i+=1

t.build(len(files))
name1= name + "_ada.ann"
t.save(name1)
t2.build(len(files))
name2 = name + "_specter.ann"
t2.save(name2)

with open('index_map' + name + '.txt', 'w') as f:
    for idx, path in index_map.items():
        f.write(f'{idx}\t{path}\n')


print("Indices created :" + name1 + " , " + name2)
print("Number of files indexed: " + str(len(files)))