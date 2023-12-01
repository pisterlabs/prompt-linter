# Import required modules and classes for setting up the chat chain.
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def rag_query_runner(local_file, topic):
    # Create a TextLoader instance to load content from a local file
    loader = TextLoader(local_file)
    # Load the documents from the specified local file
    documents = loader.load()
    # Create a RecursiveCharacterTextSplitter instance with specified parameters
    # - chunk_size: Maximum length of each text chunk
    # - chunk_overlap: Overlapping characters between adjacent chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # Split the loaded documents into smaller text chunks using the text splitter
    texts = text_splitter.split_documents(documents)
    # Define a template for chat prompts with context and questions.
    # Define the directory where the embeddings will be stored (optional)
    persist_directory = 'tempdir/db'
    # Initialize the OpenAIEmbeddings object to obtain text embeddings
    embedding = OpenAIEmbeddings()
    # Create a Chroma vector store from the provided texts using the specified embedding
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    # The embeddings of the texts are now stored in vectordb
    # Convert the vectordb into a retriever for text retrieval
    retriever = vectordb.as_retriever()
    template = """You are an intelligent assistant. Answer the question based on the following available  
    information, If you don't know just say, information unknown: \
    
    {context}
    
    Question: {question}
    
    """

    # Create a chat prompt template from the defined template.
    prompt = ChatPromptTemplate.from_template(template)

    # Create a chat model using the ChatOpenAI class.
    model = ChatOpenAI()

    # Define a function to format a list of documents as a single string.
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # Define the chat chain with multiple steps.
    chain = (
        # Step 1: Retrieve and format documents from a retriever.
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )
    # Initialize an empty report string
    report = ""
    questions = vulnerability_questions(TOPIC=topic)
    # Iterate through a list of questions
    for question in questions:
        # Invoke a function called 'chain' to process the question and obtain a response
        response = chain.invoke(question)
        # Append the response to the report with two newline characters for separation
        report += '\n\n' + f"question: {question}\nanswer: {response}"
    return report


def vulnerability_questions(TOPIC):
    questions = [
        f"What is {TOPIC}",
        f"Describe {TOPIC} vulnerability details",
        f"Who discovered recommended or acknowledged the {TOPIC} vulnerability?",
        f"What triggers {TOPIC} vulnerability or specify technical details like location, bug or mechanism?",
        f"What makes {TOPIC} vulnerability critical?",
        f"Can this {TOPIC} vulnerability triggered remotely or attack delivery methods?",
        f"What is the {TOPIC} Common Weakness Enumeration CVE, CWE or IDs for this vulnerability?",
        f"Is there a Common Vulnerabilities and Exposures (CVE) ID associated with this {TOPIC}?",
        f"What is impact or criticality of the {TOPIC} vulnerability?",
        f"What products are affected by {TOPIC} vulnerability?",
        f"Internal application is affected with {TOPIC} vulnerability, is it still severe ?",
        f"What if my application is exposing with {TOPIC} vulnerability to Internet, whatis severity ?",
        f"Have there been any known exploits of this {TOPIC}? Is there a proof-of-concept (POC) available?",
        f"Is this {TOPIC} vulnerability actively being exploited 'in the wild'?",
        f"Is there evidence about hacking or APT groups exploiting {TOPIC} vulnerability? or any related TTPs",
        f"Is there public exploit code or POC for {TOPIC} vulnerability?",
        f"Are there any details about the vendor's response to this {TOPIC} vulnerability?",
        f"Has a patch or security update been released to address this {TOPIC} vulnerability?",
        f"What few actionable steps for me as security expert to address {TOPIC} vulnerability?",
        f"What are the recommended mitigation strategies for addressing this {TOPIC} vulnerability?",
        f"What are the short-term actions that can be taken to secure systems against this {TOPIC} vulnerability?",
        f"What are the long-term strategies for enhancing the security posture and preventing {TOPIC} vulnerabilities?",
        f"What is the overall security strategy for dealing with {TOPIC} vulnerabilities?",
        f"Give some references links related to blogs {TOPIC} vulnerability"
    ]
    return questions
