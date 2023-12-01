import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

# Load environment variables from .env file
load_dotenv()
api_key = os.environ["OPEN_AI_KEY"]


def chroma_embedding(question):
    # start_time = time.time()
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", api_key)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    loader = PyPDFLoader(f"pdf_data/mental_health.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(data)

    docsearch = Chroma.from_documents(texts, embeddings)
    chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=700),
        chain_type="stuff",
        retriever=docsearch.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
    )
    # query = "The document contains the repository along with the codes. Now you need to decide which Repository contains the most complex code. Tell the name of the repository. Also explain in 100 words why you think the repository is the most complex"
    query = f"""You are a mental medical assistant. You will be given certain conditions. Your tasks will be to search over the document provided to you and provide a good solution to the mental health problem that I am facing. Use the pdf provided to give the reponse mention the names of the therapy you are suggesgting from the pdf. Give your response in a very friendly manner as if you are my best friend.

    I am facing the following problem: 

    {question}
    
    Give your answer in following formats:

    1. What is the possible name of the discorder or issue I am facing.  
    2. The explanation of the cause and effect of the discorder i am facing.
    3. The possible solutions and therapy. Describe the therapy in pointwise form too. For example like the following: 
       (a). therapy a
       (b). therapy b 
       etc. 
    4. The day to day steps I should take to cure it. 
    5. The medicines that would be best for me. 

    The points of (a), (b) etc should be in the following format: '(a).', '(b).'

    At last summarize in a very friendly manner to cheer me up. Motivate me to lead a healthy life.

    Do not give any offensive answer or that can be harmful, socially or mentally incorrect to tell.

    """
    # Run the first query.
    response = chain.run(query)

    print(response)

    return response
