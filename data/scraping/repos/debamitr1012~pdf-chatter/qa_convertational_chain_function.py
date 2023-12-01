from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.document_loaders import PyPDFLoader 
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
def qa_convertational_chain_function(
        file,
        llm,
        chain_type,
        retreiver_search_type,
        retreiver_k,
        lang
        ):
    """
    Generates a ConversationalRetrievalChain object for question-answering based on input parameters.
    This function constructs and configures a ConversationalRetrievalChain for question-answering using 
    the provided parameters. The chain is designed to generate relevant answers based on the given context 
    and questions.
    Parameters:
        file (str): The path to the PDF file for building a vector database.
        llm (LLM): An instance of the Large Language Model (LLM) for generating responses.
        chain_type (str): The type of chain to create (e.g., "stuff", "map reduce").
        retreiver_search_type (str): The search type for the retriever (e.g., "mmr").
        retreiver_k (int): The number of retrieved documents to use in the context when asking questions.
        lang (str): The prompt language
    Returns:
        ConversationalRetrievalChain: A configured ConversationalRetrievalChain object for question-answering.
    """
    # Building prompts
    if lang == "English":
        template = """Use the following snippets of context to answer the question at the end. If you don't know the answer, simply say you don't know, don't try to make up an answer.
        Answer in the same language as the language of the question.
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        follow_up_template = """Given the following conversation and a follow-up question, reformulate the follow-up question to make it an independent question, in its original language.
        conversation:
        {chat_history}
        follow-up question: {question}
        independent question:"""
    else:
        template = """Utiliza los siguientes fragmentos de contexto para responder la pregunta al final. Si no conoces la respuesta, simplemente di que no lo sabes, no trates de inventar una respuesta.
        Responde en el mismo idioma que el idioma de la pregunta.
            {context}
            Pregunta: {question}
            Respuesta útil:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        follow_up_template = """Dada la siguiente conversación y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta independiente, en su idioma original.
        conversación:
        {chat_history}
        pregunta de seguimiento: {question}
        pregunta independiente:"""
    condense_question_prompt = PromptTemplate(
        input_variables=['chat_history', 'question'],
        output_parser=None,
        partial_variables={},
        template=follow_up_template,
        template_format='f-string',
        validate_template=True
        )
    # Creating vector db
    loader = PyPDFLoader(file)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index = False,
        )
    pages = loader.load_and_split()
    chunks = text_splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
        cache_folder="./sentence_transformers"
        )
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    retreiver = vectordb.as_retriever(
        search_type=retreiver_search_type, 
        search_kwargs={"k": retreiver_k}
        )
    # Creating the chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type=chain_type, 
        retriever=retreiver,
        return_source_documents=True,
        return_generated_question=True,
        condense_question_prompt=condense_question_prompt,
        verbose = False,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain