from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from decouple import config
import os

OAI_KEY = config('OPEN_AI_API_KEY')
embeddings = OpenAIEmbeddings(openai_api_key=OAI_KEY)

small_menu_path = os.path.join('inputs', 'movies.json')
loader = TextLoader(small_menu_path)
documents = loader.load()
#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#texts = text_splitter.split_documents(documents)

#embeddings = OpenAIEmbeddings()
#docsearch = Chroma.from_documents(texts, embeddings)
docsearch = Chroma.from_documents(documents, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OAI_KEY),
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever(),
                                 verbose=True)

query = 'Please suggest some movies for a family movie night? Please include both recent and older movies?'
result = qa.run(query)
print(result)

"There are many appetizer options to choose from including edamame, seaweed salad, takoyaki, 1 pc. ebi katsu bao, crispy rice tuna, pork gyoza, 1 pc. pork kakuni bao, soft shell crab bao, tomato burrata, scallop carpaccio, maguro wasabi, sea urchin sashimi, and salmon burrata."

# https://www.lfd.uci.edu/~gohlke/pythonlibs/#annoy
# pip3 install annoy-1.17.0-cp311-cp311-win32.whl
# OR:
# pip3 install annoy-1.17.0-cp311-cp311-win_amd64.whl
# Install nemoguardrails requirements aside from annoy:
# pip3 install -r nemo_requirements.txt