from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from chainlit.types import (
    AskFileResponse
)
import chainlit as cl
import os

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

@cl.langchain_factory(use_async=True)
async def init():
    msg = cl.Message(content=f"Starting up Chainlit github chatbot...")
    await msg.send()

    docs = []
    for dirpath, dirnames, filenames in os.walk("./tmp/chainlit"):
        for file in filenames:
            try: 
                if file.endswith(".md"):
                    msg = cl.Message(content=f"Processing...{file}")
                    await msg.send()
                    loader = TextLoader(os.path.join(dirpath, file))
                    docs.extend(loader.load())
            except Exception as e: 
                pass

    split_docs = text_splitter.split_documents(docs)

    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_documents)(
        split_docs, embeddings, collection_name="github"
    )
    # Create a chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )

    # Let the user know that the system is ready
    await msg.update(content=f"Chainlit github chatbot is ready. You can now ask questions!")

    return chain


@cl.langchain_postprocess
async def process_response(res):
    answer = res["answer"]

    actions = [
        cl.Action(name="like_button", label="Like!", value="like", description="Like!"),
        cl.Action(name="dislike_button", label="Dislike!", value="dislike", description="Dislike!")
    ]

    await cl.Message(content=answer, actions=actions).send()

@cl.action_callback("like_button")
async def on_action(action):
    await cl.Message(content=f"Executed {action.name}").send()
    # Optionally remove the action button from the chatbot user interface
    await action.remove()

@cl.action_callback("dislike_button")
async def on_action(action):
    await cl.Message(content=f"Executed {action.name}").send()
    # Optionally remove the action button from the chatbot user interface
    await action.remove()