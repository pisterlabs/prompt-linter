from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings


def create_vector_db_from_youtube_url(video_url: str, openai_api_key) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(video_url, query, openai_api_key, k=4):

    llm = OpenAI(model="text-davinci-003", openai_api_key=openai_api_key)
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
            You are a helpful assistant that that can answer questions about youtube videos 
            based on the video's transcript.

            Answer the following question: {question}
            By searching the following video transcript: {docs}

            Your answers should be verbose and detailed.
            """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    db = create_vector_db_from_youtube_url(video_url, openai_api_key)
    docs = db.similarity_search(query, k)
    docs_page_content = " ".join(d.page_content for d in docs)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("/n", "")
    return response

