from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

import chainlit
from chainlit.types import AskFileResponse

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings(client=None)

welcome_message: str = """
Welcome to this Chainlit PDF Question and Answer demo! To get started:

1) Upload a PDF or text file
2) Ask a question about the file
"""


def process_file(file: AskFileResponse) -> list:
    import tempfile

    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader
    else:
        raise TypeError(
            f"""
            File is of type {file.type} and can only be text or pdf!
            """
        )

    with tempfile.NamedTemporaryFile() as tempfile:
        tempfile.write(file.content)
        loader: TextLoader | PyPDFLoader = Loader(tempfile.name)
        documents: list = loader.load()
        docs: list = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"

        return docs


def get_docsearch(file: AskFileResponse):
    docs: list = process_file(file)

    # Save data in the user session
    chainlit.user_session.set("docs", docs)
    docsearch = Chroma.from_documents(docs, embeddings)

    return docsearch


@chainlit.on_chat_start
async def start() -> None:
    # Sending an image with the local file path
    await chainlit.Message(content="You can now chat with your pdf's.").send()

    files = None
    while files is None:
        files: list[AskFileResponse] | None = await chainlit.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()  # type: ignore

    file: AskFileResponse = files[0]
    message = chainlit.Message(content=f"Processing `{file.name}`...")
    await message.send()

    docsearch = await chainlit.make_async(get_docsearch)(file)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(model="gpt-4", temperature=1, streaming=True, max_tokens=2000),
        chain_type="stuff",
        retriever=docsearch.as_retriever(max_token_limit=4097),
    )

    message.content = f"`{file.name}` processed.  You can now ask questions!"
    await message.update()

    chainlit.user_session.set("chain", chain)


@chainlit.on_message
async def main(message):
    chain: LLMChain = chainlit.user_session.get("chain")  # type: ignore
    callback = chainlit.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    callback.answer_reached = True
    result: dict = await chain.acall(message, callbacks=[callback])

    answer: str = result["answer"]
    sources: str = result["sources"].strip()
    source_elements: list = []

    documents: list = chainlit.user_session.get("docs")  # type: ignore
    metadatas: list = [doc.metadata for doc in documents]
    all_sources: list = [metadata["source"] for metadata in metadatas]

    if sources:
        found_sources: list = []

        for source in sources.split(","):
            source_name: str = source.strip().replace(".", "")

            try:
                index: int = all_sources.index(source_name)
            except ValueError:
                continue
            text = documents[index].page_content
            found_sources.append(source_name)

            source_elements.append(chainlit.Text(content=text, name=source_name))

            if found_sources:
                answer += f"\nSources: {', '.join(found_sources)}"
            else:
                answer += "\nNo sources found"

    if callback.has_streamed_final_answer:
        callback.final_stream.elements = source_elements  # type: ignore
        await callback.final_stream.update()  # type: ignore
    else:
        await chainlit.Message(content=answer, elements=source_elements).send()
