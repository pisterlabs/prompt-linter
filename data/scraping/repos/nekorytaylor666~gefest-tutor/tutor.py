from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


from config import OPENAI_API_KEY

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k",
                 openai_api_key=OPENAI_API_KEY, max_tokens=4000)


loader = DirectoryLoader("./lectures", glob="*.txt", loader_cls=TextLoader)

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY)
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())


async def ask(question):
    query_template = """
        Ты ИИ консультант для Академии Гефест. Если ты ответа нет в документ, скажи, что нужно спросить реального человека в дискорд сообществе. 
        Вопрос: {}
    """
    query = query_template.format(question)
    answer = qa.run(query)
    return answer
