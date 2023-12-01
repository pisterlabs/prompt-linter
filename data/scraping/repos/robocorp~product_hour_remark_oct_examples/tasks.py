from ast import List
from typing import Iterator
import pypdf
from robocorp.tasks import task
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.schema import Document
from langchain.schema.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from robocorp.log import console_message
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)


load_dotenv()


@task
def run_llm():
    llm = OpenAI()
    start = """
A long, long time ago
In a galaxy
""".lstrip()
    _write(start)
    _write_stream_to_console(llm.stream(start, stop="\n"))


@task
def run_chat_llm_and_see_response():
    chat = ChatOpenAI()
    messages: List[BaseMessage] = [
        SystemMessage(content="You're Batman."),
        HumanMessage(content="Say hello to the audience of my webinar."),
    ]
    chat.generate([messages])


@task
def run_chat_interactive():
    chat = ChatOpenAI()
    messages: List[BaseMessage] = [SystemMessage(content="You're a helpful assistant.")]
    while True:
        _write("Human > ")
        human_message = input()
        if not human_message.strip():
            break
        messages.append(HumanMessage(content=human_message))
        _write("AI > ")
        result = _write_stream_to_console(c.content for c in chat.stream(messages))
        messages.append(AIMessage(content=result))
    _write("\n<END OF DISCUSSION>\n")


@task
def put_the_doc_in_the_prompt():
    source = "ICC-Men-s-CWC23-Playing-Conditions-single-pages.pdf"
    reader = pypdf.PdfReader(source)
    instructions = "\n".join(p.extract_text() for p in reader.pages)
    chat = ChatOpenAI()
    messages: List[BaseMessage] = [
        SystemMessage(
            content=f"""
You're a helpful assistant. Here are the instructions to help you answer user questions:
{instructions}
""".lstrip()
        ),
        HumanMessage(content="What is Umpire?"),
    ]
    chat.generate([messages])


@task
def load_document_embeddings_to_database():
    source = "ICC-Men-s-CWC23-Playing-Conditions-single-pages.pdf"
    reader = pypdf.PdfReader(source)
    docs = (
        Document(
            page_content=page.extract_text(),
            metadata={
                "source": source,
                "page": page_number,
            },
        )
        for page_number, page in enumerate(reader.pages, start=1)
    )
    text_splitter = RecursiveCharacterTextSplitter()
    splitted_documents = text_splitter.split_documents(docs)
    db = Chroma.from_documents(
        splitted_documents, OpenAIEmbeddings(), persist_directory=".chroma"
    )
    # query it
    query = "Duckworth-Lewis-Stern table"
    data = db.similarity_search(query)

    # print results
    data[0].page_content


@task
def rag_bot():
    verbose = False
    db = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=".chroma")
    answering_chain = load_qa_chain(
        llm=ChatOpenAI(model="gpt-3.5-turbo-16k"),
        prompt=_answering_prompt(),
        verbose=verbose,
        document_prompt=_document_prompt(),
    )
    retriever_chain = LLMChain(
        llm=ChatOpenAI(), prompt=_data_query_generating_prompt(), verbose=verbose
    )
    chain = ConversationalRetrievalChain(
        combine_docs_chain=answering_chain,
        retriever=db.as_retriever(search_kwargs={"k": 10}),
        question_generator=retriever_chain,
        verbose=verbose,
    )
    messages: List[BaseMessage] = []
    while True:
        _write("Human > ")
        human_message = input()
        if not human_message.strip():
            break
        _write("AI > ")
        result = chain({"question": human_message, "chat_history": messages})
        _write(f"{result['answer']}\n")
        messages.append(HumanMessage(content=human_message))
        messages.append(AIMessage(content=result["answer"]))
    _write("\n<END OF DISCUSSION>\n")


def _document_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(
        "Content source: {source}\npage: {page}\n{page_content}\n"
    )


def _answering_prompt() -> ChatPromptTemplate:
    system_template = """
For the question provided, consider the context pieces listed in the Content section for generating your answer, but only if they seem relevant.
Your answer should rely exclusively on the information provided in the Content section.
Clearly state if the Content section does not contain sufficient information to formulate a response to the question.
Ensure that your answer is clear, coherent, and accurately represents the information from the Content.

----------------
#Content section:

{context}
----------------
Remember to direct address the User's question. Avoid unnecessary repetition and keep answers concise and compact.
Default to use the natural language the User is using in the answer.

For easy user cross-verification, include inline citations using the modified APA approach, 
placing the Content source and page number in parentheses immediately after the referenced information or at the end of each paragraph.
""".lstrip()
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    return ChatPromptTemplate.from_messages(messages)


def _data_query_generating_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(
        """
You are a content gathering assistant for ICC Men's Cricket World Cup 2023 information retrieval.
Combine the chat history and follow up question into
a standalone question. Chat History: {chat_history}
Follow up question: {question}
""".lstrip()
    )


def _write_stream_to_console(stream: Iterator[str]) -> str:
    full_message: list[str] = []
    for chunk in stream:
        _write(chunk)
        full_message.append(chunk)
    _write("\n")
    return "".join(full_message)


def _write(message: str):
    console_message(message, "stdout", flush=True)
