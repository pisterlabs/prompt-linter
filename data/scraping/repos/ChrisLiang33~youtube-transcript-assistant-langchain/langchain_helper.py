from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

video_url = 'https://www.youtube.com/watch?v=AcdZaXYh2f0'

def create_vectore_db_from_youtube_url(video_url) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db 

def get_response_from_query(db, query,k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = ' '.join([d.page_content for d in docs])

    llm = OpenAI(model='text-davinci-003')

    prompt = PromptTemplate(
        input_variables=['question', docs],
        template="""
        you are a helpful youtube assistant that can answer questions about videos based on
        the vidoe's transcript

        answer the following question: {question}
        by sarching the following video transcript: {docs}

        only use the fctual informaation form the transcript to answer the question

        if you feel likee you dont have enough information to answer the question,
        say 'i dont know'

        your answer should be detailed
        """)

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question = query, docs = docs_page_content)
    response = response.replace('\n', '')
    return response