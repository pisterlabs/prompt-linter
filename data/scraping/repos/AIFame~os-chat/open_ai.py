from icecream import ic
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

from config.constants import INDEX_NAME, OPENAI_CHAT_MODEL, OPENAI_EMBEDDINGS_LLM
from database import pinecone_db
from utils.inputs.get_repo import get_github_docs


def get_text_chunk():
    # use text_splitter to split it into documents list
    sources = get_github_docs('mertbozkir', 'docs-example')
    source_chunks = []

    md_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=1024, chunk_overlap=0,
    )

    for source in sources:
        for chunk in md_splitter.split_text(source.page_content):
            source_chunks.append(
                Document(page_content=chunk, metadata=source.metadata),
            )

    print(
        f'source_chunks length is {len(source_chunks)} and type of each source_chunks is {type(source_chunks[0])}',
    )

    return source_chunks


def upsert(data) -> Pinecone:
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDINGS_LLM)

    #   will not to use vector in memory today.
    #    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    pinecone_db.create_index(INDEX_NAME)
    # to get more information, you can look at this page
    # https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/pinecone

    vectorstore = pinecone_db.insert(
        data,
        embeddings,
    )
    return vectorstore


def create_or_get_conversation_chain(vectorstore):
    template = """
        Return results as markdown code?
    """
    #llm = ChatOpenAI(model=OPENAI_CHAT_MODEL)
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True,
    )
    prompt_template = PromptTemplate.from_template(template)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        condense_question_prompt=prompt_template,
    )
    # ic(f'conversation_chain is {conversation_chain}')
    return conversation_chain
