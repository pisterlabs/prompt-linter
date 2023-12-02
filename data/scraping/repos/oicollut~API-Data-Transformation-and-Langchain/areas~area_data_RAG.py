
from google.oauth2.credentials import Credentials
from all_funcs import initialize_google_sheet


db = initialize_google_sheet(sheet_id='1ZA9WVAAhHpmf5ikwPv1W3k1CqYNGY_Zatu5MZTrjkcg', sheet_name='Sheet1') # access Google Sheet with source websites with Google's OAuth2

# Load documents
from langchain.document_loaders import WebBaseLoader

for entry in db:
    splits = []
    for source in entry['Sources'].split(","):
        print('Loading source', source)
        loader = WebBaseLoader(source)
# Split documents
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 100)
        split = text_splitter.split_documents(loader.load())
        splits += split

    # Embed and store splits
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings
    vectorstore = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    from langchain.schema.runnable import RunnablePassthrough

    topic = "Almaden Quicksilver County Park"
    docs = vectorstore.similarity_search(topic)
    print(len(docs))

    from langchain.prompts import PromptTemplate

    """Use the following pieces of context to write an article about a US protected area for an online backpacking encyclopedia.
    The area name is given at the end. Write ten sentences maximum. Use simple, encyclopedic language. 
    {context}
    Request: {request}
    Helpful output:"""

    template =  """Use the following pieces of context to write an article about a US protected area for an online backpacking encyclopedia.
    The area name is given at the end. Use the following structure: location, what makes the area famous (or why it's protected),
    natural features and geology, recreational opportunities and available infrastructure. No concluding paragraph.
    If you don't know about a topic, don't make anything up, but rather skip the topic completely (don't invent, don't admit you don't know).
    Write ten sentences maximum. Use simple, encyclopedic language. 
    {context}
    Request: {request}
    Helpful output:"""
    rag_prompt_custom = PromptTemplate.from_template(template)

    
    rag_chain = (
        {"context": retriever, "request": RunnablePassthrough()} 
        | rag_prompt_custom 
        | llm 
        
    )
    
    print(rag_chain.invoke("Almaden Quicksilver State Park: location, what makes it famous (or why it's protected), natural features and geology, recreational opportunities and infrastructure"))
    break
