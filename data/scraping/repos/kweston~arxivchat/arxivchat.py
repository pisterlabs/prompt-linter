import argparse
import chromadb
import gradio as gr
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseRetriever, Document
from langchain.agents import initialize_agent, Tool, ConversationalChatAgent
from langchain.agents import AgentType, AgentExecutor
from langchain.schema import OutputParserException
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import StdOutCallbackHandler
from arxiv_chat.arxivpdf import ArxivPDF
from typing import List
from pathlib import Path
from typing import Dict, Any, Optional, List
from uuid import UUID

# langchain.debug = True

CHROMA_DB_DIR = "./chroma_db"


class MyCustomHandler(BaseCallbackHandler):
    def on_tool_start(self, token: str, **kwargs) -> None:
        print(f"My custom handler, token: {token}")
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool starts running."""
        pass


def format_front_matter(abstract_metadata: dict) -> Document:
    """
    Format the front matter of an arxiv paper into a document that can be
    be understood by the RAG
    """
    out = ""
    for k, v in abstract_metadata.items():
        out += f"{k}: {v}\n\n"
    return Document(page_content=out, metadata=abstract_metadata)


def create_vector_store(
        docs: List[Document], collection_name: str, force_overwrite:bool=True
    )-> Chroma:
    """
    Create a vectorstore from a list of documents
    """
    embeddings_obj = OpenAIEmbeddings()
    embedding_function = embeddings_obj.embed_documents
    persistent_client = chromadb.PersistentClient(CHROMA_DB_DIR)
    collections = set([col.name for col in persistent_client.list_collections()])
    print(f"Existing collections: {collections}")

    if not force_overwrite and collection_name in collections:
        print(f"Loading {collection_name} from disk")
        # load from disk
        collection = persistent_client.get_collection(collection_name)
        vectorstore = Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=embeddings_obj,
        )
    else:
        if force_overwrite:
            print(f"Creating {collection_name} and saving to disk")
            persistent_client.delete_collection(collection_name)
        # create and save to disk
        collection = persistent_client.create_collection(
            collection_name,
            embedding_function=embedding_function,
        )

        collection.add(
            ids=[str(i) for i in range(len(docs))],
            documents=[doc.page_content for doc in docs],
            # metadatas=[doc.metadata for doc in all_splits],
        )

        vectorstore = Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=embeddings_obj,
        )
    return vectorstore


def create_docQA_chain(fname: str, force_overwrite:bool=True):
    """
    Create a RetrievalQA chain from a pdf file
    """
    pdf = ArxivPDF()
    front_matter, body, doc_file_name = pdf.load(query=fname, parse_pdf=True, split_sections=False, keep_pdf=True)
    header = format_front_matter(front_matter[0].metadata)
    docs = [header] + body

    # Define our text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(docs)

    vectorstore = create_vector_store(all_splits, fname, force_overwrite)

    llm = OpenAI(temperature=0, verbose=True)
    qa_chain =  RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        verbose=True,
        return_source_documents=False,
    )
    return qa_chain, front_matter[0].metadata


def main(
        fnames: List[str],
        force_overwrite:bool=True,
        questions:Optional[List[str]] = None,
        no_tools: bool=False,
        no_gui: bool=False,
        verbose: bool=False
):
    tools = []
    if not no_tools:
        for fname in fnames:
            qa_chain, metadata = create_docQA_chain(fname, force_overwrite)
            tool = Tool(
                name=fname,
                func=qa_chain,

                description=f"""
                useful for when you need to answer questions about the paper titled {metadata['Title']}. Input should be a fully formed question.
                """
            )
            print(tool.description)
            tools.append(tool)

    llm = ChatOpenAI(temperature=0, verbose=True)
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    PREFIX = """
    I am a conversational chat agent that can answer questions about papers on arxiv. 
    I can also answer other general questions.
    """
    SUFFIX = """Begin!
    {chat_history}
    Question: {input}
    Thought:{agent_scratchpad}"""

    agent: AgentExecutor = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        # early_stopping_method='generate',
        max_iterations=3,
        # agent_kwargs={
        #     'prefix': PREFIX,
        #     'suffix': SUFFIX,
        # }
    )

    """
    TODO: 
    reorder prompt messages so that the chat history is before the scratchpad
    before
    0 System Prompt
    1 Chat history
    2 Human message
    3 Agent scratchpad

    0 System Prompt
    1 Human message
    2 Chat history
    3 Agent scratchpad

    #agent.agent.llm_chain.prompt.messages = [messages[0], messages[2], messages[1], messages[3]]
    """

    # modify the prompt suffix of this agent to work with memory
    handlers = [StdOutCallbackHandler(), MyCustomHandler()]

    def _cleanup_response(e: Exception) -> str:
        print("Warning: Could not parse LLM output")
        response = str(e)
        prefix = "Could not parse LLM output: "
        if response.startswith(prefix):
            return response.removeprefix(prefix).removesuffix("`")
        else:
            raise(e)
        
    def _run_agent(agent: AgentExecutor, question: str) -> str:
        try:
            if verbose:
                answer = agent.run(input=question, callbacks=handlers)
            else:
                answer = agent.run(input=question)
        except OutputParserException as e:
            answer = _cleanup_response(e)
        return answer

    if questions is None:
        if no_gui:
            while True:
                question = input("Enter a question (q to quit): ")
                if question.strip() == "q":
                    break
                answer = _run_agent(agent, question)
                print(answer)
        else:
            def llm_response(question, history=None):
                answer = _run_agent(agent, question)
                return answer 

            gr.ChatInterface(llm_response).launch()
    else:
        for question in questions:
            print(f"query: {question}")
            out = _run_agent(agent, question)
            print(f"result: {out}")
            print()


def parse_cli_args(args: Optional[List[str]]=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run doc QA on the given Arxiv PDF",
        usage="python run.py --fname 2302.0092"
    )
    parser.add_argument(
        "--fname",
        type=str, help="The Arxiv ID of the paper to run QA on",
        default="2307.09288",
        # Restrict to the following two papers for now until parsing is more robust
        choices = [
            "2302.00923", # Multimodal CoT Reasoning paper
            "2211.11559", # Visual programming paper
            "2307.09288", # llama 2 paper
        ]
    )
    parser.add_argument("--force_overwrite", "-f", action="store_true", help="Force overwrite of existing Chroma DB")
    parser.add_argument("--live_input", action="store_true", help="Live input mode")
    parser.add_argument(
        "--questions", "-q",
         help="""A text file containing questions to ask the agent. If specified, we will not run 
                in live input mode.""",
        default=None,
        type=Path
    )
    parser.add_argument("--no_tools", action="store_true", help="Don't load any tools. Useful for debugging")
    parser.add_argument("--no_gui", action="store_true", help="Don't load the GUI")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_cli_args()
    if args.questions is not None:
        with open(args.questions, 'r') as f:
            questions = f.readlines()
    else:
        questions = None

    print(f"Running with args: {args}")

    main([args.fname], args.force_overwrite, questions, args.no_tools, args.no_gui, args.verbose)
