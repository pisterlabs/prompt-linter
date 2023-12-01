import dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

dotenv.load_dotenv()

from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate

from langchain.document_loaders import TextLoader

embedding_model = OpenAIEmbeddings()

llm = ChatOpenAI(temperature=0)

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


loaders = [
    TextLoader('langchain_blog_posts/blog.langchain.dev_announcing-langsmith_.txt'),
    TextLoader('langchain_blog_posts/blog.langchain.dev_benchmarking-question-answering-over-csv-data_.txt'),
    TextLoader('langchain_blog_posts/blog.langchain.dev_chat-loaders-finetune-a-chatmodel-in-your-voice_.txt'),
]
docs = []

for l in loaders:
    docs.extend(l.load())

print(len(docs))

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

prompt_template = """Please answer the user's question as related to Large Language Models
Question: {question}
Answer:"""

prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

llm_chain = LLMChain(llm=llm, prompt=prompt)

embeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=embedding_model
)

docsearch = Chroma.from_documents(texts, embeddings)

query = "What are chat loaders?"
docs = docsearch.similarity_search(query)
print(docs[0].page_content)
