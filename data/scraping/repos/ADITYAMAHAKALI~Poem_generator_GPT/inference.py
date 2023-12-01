from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.llms import OpenAI
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain

# Load the .env file
load_dotenv()
def generate_poems(topic,search_index):
    docs = search_index.similarity_search(topic, k=4)
    inputs = [{"context": doc.page_content, "topic": topic} for doc in docs]
    return chain.apply(inputs)[0]['text']
# Get the API key
api_key = os.getenv("OPENAI_API_KEY")
persist_directory = 'db'
prompt_template = """Use the context below to write a 8 stanzas poem about {topic} in the style of context.
    Context: {context}
    Topic: {topic}
    Poem:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "topic"]
)

llm = OpenAI(temperature=0,openai_api_key=api_key)
chain = LLMChain(llm=llm, prompt=PROMPT)
embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

print(generate_poems("India",vectordb))

