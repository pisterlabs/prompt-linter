# Std libs
from datetime import datetime
from operator import itemgetter

# Third Party libs
from dotenv import load_dotenv

# LangChain libs
from langchain.document_loaders import RecursiveUrlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser


class AssistantChain:
    """AssistantChain
    Based on a user's prompt, this will then prompt gpt-3.5-turbo-16k
    and return the response from the llm.
    """

    def __init__(self, name):
        # Load environment (API keys)
        load_dotenv()

        # Define chain_1
        template = f"You are a helpful assistant who's name is {name}."
        human_template = "{question}"

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template),
                ("human", human_template),
            ]
        )
        self.chain = (
            chat_prompt
            | ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
            | StrOutputParser()
        )

    def get_chain(self):
        return self.chain

class DocumentationChain:
    """DocumentationChain
    Based on a user's prompt, this will retrieve the most similar chunks
    of text from the vector store (using retrievers) and add this information
    to the prompt, before supplying the prompt to the language model.
    """

    def __init__(self, url):
        # Load environment (API keys)
        load_dotenv()

        # Load in langsmith documentation as test
        api_loader = RecursiveUrlLoader(url)
        raw_documents = api_loader.load()

        # Transformer
        doc_transformer = Html2TextTransformer()
        transformed = doc_transformer.transform_documents(raw_documents)

        # Splitter
        text_splitter = TokenTextSplitter(
            model_name="gpt-3.5-turbo",
            chunk_size=2000,
            chunk_overlap=200,
        )
        documents = text_splitter.split_documents(transformed)

        # Define vector store based
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # Define chain_2
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful documentation Q&A assistant, trained to answer"
                    " questions from LangSmith's documentation."
                    " LangChain is a framework for building applications using large language models."
                    "\nThe current time is {time}.\n\nRelevant documents will be retrieved in the following messages.",
                ),
                ("system", "{context}"),
                ("human", "{question}"),
            ]
        ).partial(time=str(datetime.now()))

        model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
        response_generator = prompt | model | StrOutputParser()

        self.chain = (
            # The runnable map here routes the original inputs to a context and
            # a question dictionary to pass to the response generator.
            {
                "context": itemgetter("question")
                | retriever
                | (lambda docs: "\n".join([doc.page_content for doc in docs])),
                "question": itemgetter("question"),
            }
            | response_generator
        )

    def get_chain(self):
        return self.chain
