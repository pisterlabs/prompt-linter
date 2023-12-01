
#from all_funcs import initialize_google_sheet

from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough

#db = initialize_google_sheet(sheet_id='1ZA9WVAAhHpmf5ikwPv1W3k1CqYNGY_Zatu5MZTrjkcg', sheet_name='Sheet1') # access Google Sheet with source websites with Google's OAuth2
#db = ["https://destinationlancasterca.org/adventures/antelope-valley-california-poppy-reserve/", "https://www.visitcalifornia.com/experience/antelope-valley-california-poppy-reserve/", "https://modernhiker.com/hike/hiking-the-antelope-valley-california-poppy-preserve/"]
# Load documents
db = ["https://en.wikipedia.org/wiki/Ancient_Bristlecone_Pine_Forest"]
#db = ["https://sierranevadageotourism.org/entries/ancient-bristlecone-pine-forest/70b068b8-0b52-4038-b765-dbc817a263b0", "https://bishopvisitor.com/activities/bristlecone-forest/", "https://www.visitmammoth.com/trip-ideas/exploring-ancient-bristlecone-pine-forest/", "https://roadtrippingcalifornia.com/ancient-bristlecone-pine-forest/"]
from langchain.document_loaders import WebBaseLoader

for source in db:
    splits = []
    #for source in entry['Sources'].split(","):
    print('Loading source', source)
    loader = WebBaseLoader(source)
# Split documents
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    split = text_splitter.split_documents(loader.load())
    splits += split
    # Embed and store splits
vectorstore = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings(), persist_directory="./chroma_db_ancient_bristlecone_500_WIKI")

llm = ChatOpenAI(model_name="gpt-4", temperature=1)
"gpt-3.5-turbo"

vectorstore = Chroma(persist_directory="./chroma_db_ancient_bristlecone_500_WIKI", embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={'k': 6})

#query = "nature of the park"
#query = "history, past, year, archaeology, indigenous people, century, originally, former, previous"

#doc = vectorstore.similarity_search(k=1, query=query)
#docs = retriever.get_relevant_documents(query)

#vectorstore.get(include=)


#for doc in docs:
    #print(doc)
    #print("\n")
#exit


template =  """Use the following pieces of context to provide users with information about nature in the Ancient Bristlecone Pine Forest.
If you don't know something, don't make it up. Write two sentences maximum. Use simple, encyclopedic language.
{context}
Helpful output:"""
rag_prompt_custom = PromptTemplate.from_template(template)


rag_chain = (
        {"context": retriever} 
        | rag_prompt_custom 
        | llm 
        
    )

print(rag_chain.invoke("nature and vegetation of this land"))

