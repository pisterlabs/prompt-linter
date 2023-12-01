from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader, CSVLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import openai
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from send_email import message_send
embeddings = OpenAIEmbeddings(
    openai_api_key="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")
embeddings = OpenAIEmbeddings(
    openai_api_key="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")

openai.api_key = "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"


PINECONE_API_KEY = '2f1f9a16-8e97-4485-b643-bbcd3618570a'
PINECONE_ENVIRONMENT = 'us-west1-gcp-free'

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=0)


system_instructions = """

You are an online shopping assistant and you have to help your customer to find the right product.
You are going to be given a series of products, along with a description of a customer's profile.
For each product, you will be inputted a description and a price.
Tailor your responses to the customer's profile and make appropriate recommendations.
If they give pricing constraints, do not go over the pricing constraints.
If they give a product category, try to recommend a product in that category.
If a product does not exist or is out of stock, do not recommend it.

Furthermore, when you are recommending a product, you should give a reason why you are recommending it.
For example, if the customer is looking for a pair of shoes, you should say something like "These shoes are very comfortable and are made of high quality materials."

Additionally, we will also be feeding you a series of products that you want to recommend to the customer. These are high margine products so we want to sell them. However, if it doesn't make sense to recommend these products, do not recommend them.

Make sure you are using the customer's profile to make suggestions - cite the customer's profile if appropriate when making suggestions. Explain what about their profile made you recommend that product to them.

"""


def getLoader(type, file_path):
    if type == "csv":
        return CSVLoader(file_path)
    elif type == "pdf":
        return PyPDFLoader(file_path)
    else:
        return OnlinePDFLoader(file_path)


def getPrompt(query, user_profile):
    query = f"""
    {system_instructions}

    Here is the user's profile:
    {user_profile}

    Here are some types of products that we want to promote:
    We want to promote blue-colored clothes. 

    Here is the query:
    {query}

    Your response, if the user is asking to buy something, should be structured like:

    
    <product_1> | <price_1> | <url associated with <product_1>>
    <product_2> | <price_2> | <url associated with <product_2>>
    ...
    <reasoning for selecting these products>

    We are putting the output through a QA model, so make sure that you are outputting the product name, price, and url in the format above.
    """
    print (query)
    return query

def parseResponse(response):
    # get the urls from the response
    lines = response.split("\n")
    product_lines = []
    for line in lines:
        if "|" in line:
            product_lines.append(line)
    links = []
    for product_line in product_lines:
        links.append(product_line.split("|")[1].strip())
    return links




def getResponse(query, user_profile, docsearch):
    # get the path_name of the file in data folder
    print ("GETTING RESPONSE")
    llm = OpenAI(
        temperature=0.3, openai_api_key="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")
    chain = load_qa_chain(llm, chain_type="stuff")

    print ("Searching")
    docs = docsearch.similarity_search(query, 2)
    print ("Found")
    prompt = getPrompt(query, user_profile)
    
    result = (chain.run(input_documents=docs, question=prompt))
    print(parseResponse(result))
    print(strengthen_profile(user_profile, query + "\n" + result))
    email = (sendOutreachEmail(user_profile, query + "\n" + result, "There is a new summer sale selling all blue clothes for 50 dollars off", docsearch))
    message_send(email, "arthbohra@berkeley.edu")
    return result


def strengthen_profile(cur_profile, recent_chat):

    profile_current = "Here is the current profile (it may be empty): " + cur_profile + "\n"
  
    prompt = f"""
    We are an e-commerce platform that sells products to customers. We want to strengthen our customer profiles by adding information about the products that they have bought and conversations they have had.
    For example, if a customer has bought a product, we want to add information about that product to their profile. If a customer has had a conversation with a customer service representative, we want to add information about that conversation to their profile.
    By adding information, we don't want to just copy the direct product names into their profile - rather, we want to derive insights about the persona and background of the user.
    For example, if the user is buying hiking clothes and talking about playing sports, we can assume that this user is an active individual.
    If the user is buying a lot of books and talking about reading, we can assume that this user is an avid reader.
    If the user talks about keeping warm, the user may live in a cold area, so save that he likes to be warn and might live in a cool environment.
    Do not, under any circumstances, add anything about what products the customer has bought.
    Just add to the current user profile or edit the current user profile with trends you are noticing in the user and any additional facts about them.

    Here is the user's current profile:    
    {profile_current}

    Here is their most recent chat - this may be structured like a transcripts.
    {recent_chat}
    

    Remember, the new user profile should not include anything about the products that the user has bought - only trends.

    New user profile: 
    """

    
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt= prompt,
        temperature=0.3,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response['choices'][0]['text']



def getOutReachPrompt(customer_profile, recent_chat, promotional_updates):
    instructions = f'''You are an email sending bot that is sending an email to a customer. You are going to be given a customer profile and a recent chat transcript. You should use this information to send an email to the customer. The email should
    - be personalized to the customer. For example, if the customer is a student, you should say something like "As a student, I know that you are very busy, so I will keep this email short."
    - be personalized to the recent chat transcript. For example, if the customer is talking about a product, you should say something like "I noticed that you were talking about this product, so I wanted to send you some more information about it."

    Furthermore, we will also be highlighting any promotional updates, whether they are new products or new deals. You should include these in your email.

    Here is the customer_profile:
    {customer_profile}

    Here is the customer's recent chat transcript:
    {recent_chat}

    Here are the promotional updates:
    {promotional_updates}

    Your email must be structured like:
    
    <greeting>

    <body>

    <closing>

    Email:

    '''
    return instructions

def sendOutreachEmail(customer_profile, recent_chat, promotional_updates, docsearch):
    llm = OpenAI(
        temperature=0.3, openai_api_key="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")
    chain = load_qa_chain(llm, chain_type="stuff")

    print ("Searching")
    docs = docsearch.similarity_search(customer_profile + recent_chat + promotional_updates , 2)
    print ("Found")
    prompt = getOutReachPrompt(customer_profile, recent_chat, promotional_updates)
     
    result = (chain.run(input_documents=docs, question=prompt))
    return result

