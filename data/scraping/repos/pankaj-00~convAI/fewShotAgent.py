from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import GPT4AllEmbeddings
from langchain.document_loaders import TextLoader
from langchain.agents import initialize_agent
from langchain import FewShotPromptTemplate
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.agents import Tool
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import json
# import sys

load_dotenv()

def getPrompt():
    # create our examples
    examples = [
    {
        "query": "Hello",
        "answer": "Hello, welcome to Walmart customer service. How can I help you?"
    },
    {
        "query": "How can I order any product from the walmart?",
        "answer": "For availing our services, you can visit walmart.com on a web browser or install our app through app store or the playstore. You can select a product you want to order and and click the buy now button below the product description and then follow the instructions specified on the app or the webpage."
    },
    {
        "query": "My order was not delivered. Can you help me track it?",
        "answer": "Oh we are so sorry for the inconvenience. Can you give me the order number so I can help you track the order."
    },
    {
        "query": "What is the process for returning an item to Walmart?",
        "answer": "You can return an item by checking the options tab under 'Your orders' on the walmart website or app and then opting for 'return an item'."
    },
    {
        "query": "What is 2+2?",
        "answer": "Please ask questions related to walmart only."
    },
    ]


    # create a example template
    example_template = """
    User: {query}
    AI: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """You are a courteous customer service agent for Walmart, and your 
    responses should be concise and directly related to Walmart-related queries. Be
    as conversational as possible with the user. If a user asks a question unrelated to Walmart, kindly remind them to inquire 
    about Walmart. Utilize additional tools only when essential and provide responses 
    in a single sentence. Ensure that all responses pertain to Walmart. Below are some example conversations between the assistant and the customer: 
    """

    # and the suffix our user input and output indicator
    suffix = """
    User: {query}
    AI: """

    # now create the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n"
    )
    
    return few_shot_prompt_template

def getDB():
    INDEX_DIR = os.environ.get('INDEX_DIR')
    return {
        "searchDB": FAISS.load_local(os.getcwd() + "/services/data/walmart_index", gpt4allemb),
        "productDB": FAISS.load_local(os.getcwd() + "/services/data/product_index", gpt4allemb),
        "policiesDB": FAISS.load_local(os.getcwd() + "/services/data/policies_index", gpt4allemb)
    }

def getconvAgent():
    def walmartSearch(query: str) -> str:
        docs = searchDB.similarity_search(query, 2)
        return docs[0].page_content+docs[1].page_content

    def walmartProduct(query: str) -> str:
        docs = productDB.similarity_search(query, 2)
        return docs[0].page_content+docs[1].page_content

    def walmartPolicies(query: str) -> str:
        docs = policiesDB.similarity_search(query, 2)
        return docs[0].page_content

    # Set of tools for the conversational agent.
    tools = [
        Tool(
            name="Walmart Information",
            func=walmartSearch,
            description='Use this tool to get information related Walmart as a company.'
        ),
        Tool(
            name="Walmart Products",
            func=walmartProduct,
            description="Use this tool to get information related to the products sold by Walmart."
        ),
        Tool(
            name="Walmart Policies",
            func=walmartPolicies,
            description="Use this tool to get information related to the return policies, coupon policies and price-match policies."
        ),
    ]
    conversationalAgent = initialize_agent(
        agent='zero-shot-react-description', 
        tools=tools, 
        llm=llm,
        max_iterations=3,
        handle_parsing_errors=True,
        verbose = True
    )
    
    return conversationalAgent

#Instantiation of OpenAI model
OPEN_AI_API_KEY = os.environ.get('OPEN_AI_API_KEY')
llm = OpenAI(openai_api_key = OPEN_AI_API_KEY)
print("LLM ready!")

#Instantiation of Embeddings
gpt4allemb =  GPT4AllEmbeddings()
print('gpt4allembeddings ready!')

#Instantiation of Indexes
db = getDB()
print("DB ready!")
searchDB = db['searchDB']
productDB = db['productDB']
policiesDB = db['policiesDB']


def mainFunc(query):
    try:
        convAgent = getconvAgent()
        promptTemplate = getPrompt()
        # sampleQuery = "How to place a new order?"
        result = convAgent(promptTemplate.format(query = query))
        # result = convAgent(promptTemplate.format(query = sampleQuery))
        return result
    except:
        print("FewShot Agent Error")   

# mainFunc()