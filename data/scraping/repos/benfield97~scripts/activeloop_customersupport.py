import os 
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import SeleniumURLLoader
from langchain import PromptTemplate
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ['ACTIVELOOP_TOKEN'] = os.getenv('ACTIVELOOP_TOKEN')

urls = ['https://beebom.com/what-is-nft-explained/',
        'https://beebom.com/how-delete-spotify-account/',
        'https://beebom.com/how-download-gif-twitter/',
        'https://beebom.com/how-use-chatgpt-linux-terminal/',
        'https://beebom.com/how-delete-spotify-account/',
        'https://beebom.com/how-save-instagram-story-with-music/',
        'https://beebom.com/how-install-pip-windows/',
        'https://beebom.com/how-check-disk-usage-linux/']


# use the selenium scraper to load the documents 
loader = SeleniumURLLoader(urls=urls)
docs_not_splitted = loader.load()


#split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs_not_splitted)

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

#create deeplake dataset
# my_activeloop_org_id = 'benfield'
# my_activeloop_dataset_name = 'langchain_course_from_zero_to_hero'
# dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
# db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db= FAISS.from_documents(docs, embeddings)

# add documents to our deep lake dataset
db.add_documents(docs)

# find the top relevant documents to a specific query
query = 'how to check disk usage in linux?'
docs = db.similarity_search(query)
print(docs[0].page_content)


# let's write a prompt for a customer support chatbot
# answer questions using information extracted from our db
template =  """You are an exceptional customer support chatbot that gently answer questions.

You know the following context information.

{chunks_formatted}

Answer to the following question from a customer. Use only information from the previous context information. Do not invent stuff.

Question: {query}

Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables = ['chunks_formatted', 'query'],
)

# the full pipeline

#user questions
query = 'How to check disk usage in linux?'

#retrieve relevant chunks
docs = db.similarity_search(query)
retrieved_chunks = [doc.page_content for doc in docs]

#format the prompt
chunks_formatted = "\n\n".join(retrieved_chunks)
prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

#generate the answer
llm = OpenAI(model='text-davinci-003', temperature=0)
answer = llm(prompt_formatted)
print(answer)


'''
helpful point:
Suppose we ask, "Is the Linux distribution free?" 
and provide GPT-3 with a document about kernel features as context. 
It might generate an answer like "Yes, the Linux distribution is free to 
download and use," even if such information is not present in the context 
document. Producing false information is highly undesirable for customer service 
chatbots!

GPT-3 is less likely to generate false information when the answer to the user's 
question is contained within the context. Since user questions are often brief 
and ambiguous, we cannot always rely on the semantic search step to retrieve the 
correct document. Thus, there is always a risk of generating false information.
'''