import os
import sys
import time


'''
Documentation:
https://python.langchain.com/docs/get_started/quickstart

Reference:

'''

def step1_QueryOpenAI():
    from langchain.llms import OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    llm = OpenAI(temperature=0.9)

    text = "What are the 5 vacation destinations for someone who likes to eat Samosa?"
    print(llm(text))


def step2_PromptTemplates():
    from langchain.prompts import PromptTemplate
    from langchain.llms import OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["food"],
        template="What are the 5 vacation destinations for someone who likes to eat {food}?",
    )
    print(prompt.format(food="dessert"))
    print(llm(prompt.format(food="dessert")))


def step3_Chaining():
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from dotenv import load_dotenv

    load_dotenv()

    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["food"],
        template="What are the 5 vacation destinations for someone who likes to eat {food}?",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.run(("Fruit")))


def step4_AgentsCreation_1():
    # pip install google-search-results

    from langchain.llms import OpenAI
    from langchain.agents import initialize_agent
    from langchain.agents import load_tools

    from dotenv import load_dotenv

    load_dotenv()
    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True
    )
    agent.run(
        "Who is the current Prime Minister of India? What is the largest prime number that is smaller than their age?"
    )


def step5_Memory():
    from langchain import OpenAI, ConversationChain

    from dotenv import load_dotenv

    load_dotenv()

    llm = OpenAI(temperature=0)
    conversation = ConversationChain(llm=llm, verbose=True)
    conversation.predict(input="Hi There")
    conversation.predict(input="How are you doing")
    time.sleep(5)


def step6_HackerNews_DocumentLoaders():
    from langchain.document_loaders import HNLoader

    loader = HNLoader("https://news.ycombinator.com/item?id=34422627")
    data = loader.load()
    print(f"Found {len(data)} Comments")
    print(
        f"Here are some samples: \n\n{''.join([x.page_content[:150] for x in data[:2]])}"
    )


def step7_PyPDFLoader_DocumentLoaders():

    from dotenv import load_dotenv
    load_dotenv()

    from langchain.document_loaders import PyPDFLoader
    from langchain import OpenAI

    llm = OpenAI(temperature=0)

    loader = PyPDFLoader(
        "pdfdocs\FDP Requirements.pdf"
    )
    data = loader.load()

    from langchain.chains.question_answering import load_qa_chain

    chain = load_qa_chain(llm, chain_type="map_reduce")
    query = "What is the Overview of Functional Data Platform Requirements"
    response = chain.run(input_documents=data, question=query)
    print(response)


def step8_XX_RetreivalQA_PyPDFLoader_DocumentLoaders():

    from dotenv import load_dotenv
    load_dotenv()

    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings

    from langchain import OpenAI

    llm = OpenAI(temperature=0)

    loader = PyPDFLoader("pdfdocs\SOP.pdf")
    data = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_doc = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(chunked_doc, embeddings)

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 7})
    from langchain import hub

    rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")
    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retreiver=retriever,
        chain_type_kwargs={"prompt": rag_prompt_llama},
    )

    result = qa_chain(
        {"query": "What is the Overview of Functional Data Platform Requirements?"}
    )
    print(result)


def step9_ConversationalRetreivalChain_singleDocument():
    from dotenv import load_dotenv

    load_dotenv()

    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=0)

    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains import ConversationalRetrievalChain

    embeddings = OpenAIEmbeddings()

    loader = PyPDFLoader(
        "pdfdocs\SOP.pdf"
    )
    data = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_doc = text_splitter.split_documents(data)

    vectordb = Chroma.from_documents(chunked_doc, embeddings)

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 7})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever, return_source_documents=True
    )
    chat_history = []
    import sys

    while True:
        query = input("Prompt: ")
        if query == "exit" or query == "quit" or query == "q":
            print("Exiting")
            sys.exit()
        result = qa_chain({"question": query, "chat_history": chat_history})
        print("Answer:" + result["answer"])
        chat_history.append((query, result["answer"]))


def step10_ConversationalRetreivalChain_multipleDocument():
    from dotenv import load_dotenv
    load_dotenv()

    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains import ConversationalRetrievalChain
    from langchain.document_loaders import Docx2txtLoader, TextLoader

    documents = []
    for file in os.listdir("pdfdocs"):
        print(file)
        if file.endswith(".pdf"):
            pdf_path = "./pdfdocs/" + file
            print(pdf_path)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

        elif file.endswith(".docx") or file.endswith(".doc"):
            doc_path = "./pdfdocs/" + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())

        elif file.endswith(".txt"):
            doc_path = "./pdfdocs/" + file
            loader = TextLoader(doc_path)
            documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunked_doc = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(chunked_doc, embeddings)
    vectordb.persist()
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 7})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever, return_source_documents=True
    )

    yellow = "\033[0;33m"
    green = "\033[0;32m"
    white = "\033[0;39m"

    chat_history = []
    print(f"{yellow}---------------------------------------------------------------------------------")
    print('Welcome to the DocBot. You are now ready to start interacting with your documents')
    print('---------------------------------------------------------------------------------')
    while True:
        query = input(f"{green}Prompt: ")
        if query == "exit" or query == "quit" or query == "q" or query == "f":
            print('Exiting')
            sys.exit()
        if query == '':
            continue
        result = qa_chain(
            {"question": query, "chat_history": chat_history})
        print(f"{white}Answer: " + result["answer"])
        chat_history.append((query, result["answer"]))


step10_ConversationalRetreivalChain_multipleDocument()
# step4_AgentsCreation_1()
# step5_Memory()
# step6_HackerNews_DocumentLoaders()
# step7_PyPDFLoader_DocumentLoaders()
# step8_XX_RetreivalQA_PyPDFLoader_DocumentLoaders()
# step9_ConversationalRetreivalChain_singleDocument()
# step10_ConversationalRetreivalChain_multipleDocument()
