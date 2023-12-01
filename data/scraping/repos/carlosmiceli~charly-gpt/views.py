import os
import logging
import json
import traceback
import pinecone
import openai
from django.shortcuts import render
from django.http import JsonResponse
from .helpers.index import clean_up_text, hash_url_to_id, tiktoken_len, process_and_upsert_chunks, create_full_prompt
from langchain.llms import OpenAI
from langchain.document_loaders import BrowserlessLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import LLMChain
from langchain.agents import AgentType, Tool, initialize_agent
# from langchain.utilities import SerpAPIWrapper
from langchain.chains import RetrievalQA, ConversationChain, RetrievalQAWithSourcesChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate

# Get an instance of a logger
logger = logging.getLogger(__name__)

# Initialize OpenAI
openai_api_key = os.environ.get('OPENAI_API_KEY')
llm = ChatOpenAI(openai_api_key=openai_api_key,
                 model_name='gpt-3.5-turbo', temperature=0.5)
embed = OpenAIEmbeddings(openai_api_key=openai_api_key)
max_token_limit = 4000

# Initialize Pinecone
pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment='us-west4-gcp-free')
index = pinecone.Index(os.environ.get('PINECONE_INDEX_NAME'))

# Create an instance of the SerpAPIWrapper if the API key is available
# serp_api_key = os.environ.get('SERP_API_KEY')
# serpapi = SerpAPIWrapper(serpapi_api_key=serp_api_key)


def form(request):
    return render(request, 'form.html')

def delete_url(request):
    if request.method == 'POST':
        try:
            # logger.info(f"1: {index.describe_index_stats()}")
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)
        try:
            url = data.get('url')
            url_id = str(hash_url_to_id(url))
            ids = [f"{i + 1}-{url_id}" for i in range(1000)]
            index.delete(ids=ids)
            return JsonResponse({'success': 'URL vectors deleted successfully.'}, status=200)
        except Exception as e:
            logger.error(f"Error while querying index: {str(e)}")
            logger.error(traceback.format_exc())
            return JsonResponse({'error': 'Error occurred while trying to delete URL vectors.'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method.'})


def chat_agent(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)

        try:
            # Load data from the URL
            query = data.get('input', '')
            url = data.get('url') or None
            store = data.get('store')

            if url: 
                url_id = str(hash_url_to_id(url)) or None
                ids = [f"{i + 1}{url_id}" for i in range(20)]
                logger.info(f"ids: {ids}")
                check_url_vectors = index.fetch(ids=ids)
                logger.info(f"check_url_vectors: {check_url_vectors}")
                #if url vectors already exist, set a new bool to True
                if len(check_url_vectors['vectors']) > 0:
                    url_vectors_exist = True

                logger.info(f"1: {index.describe_index_stats()}")

                embed_function = openai.Embedding.create
                query_function = index.query
                engine='text-embedding-ada-002'
                
                metadata = {
                    'url_id': url_id,
                    'url': url,
                }

                # Initialize the BrowserlessLoader
                browserless_loader = BrowserlessLoader(
                    api_token=os.environ.get('BROWSERLESS_API_KEY'),
                    urls=[url],  # Load a single URL
                )
                documents = browserless_loader.load()

                page_content = clean_up_text(documents[0].page_content)

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=300,
                    chunk_overlap=10,
                    length_function=tiktoken_len,
                    separators=["\n\n", "\n", " ", ""]
                )

                chunks = text_splitter.split_text(page_content)

                try:
                    process_and_upsert_chunks(chunks, metadata, url_id, embed_function, engine, index)
                    logger.info(f"2: {index.describe_index_stats()}")
                except:
                    logger.error(f"Error while processing and upserting chunks: {str(e)}")
                    logger.error(traceback.format_exc())
                    return JsonResponse({'error': 'Error occurred while processing and upserting chunks.'}, status=500)
                
            query_embed = embed_function(input=[query], engine=engine)

            query_response = query_function(query_embed['data'][0]['embedding'], top_k=3, include_metadata=True)

            prompt = create_full_prompt(query, query_response)

            # PICK UP HERE
            # GENERATIVE ANSWER
            # FULL MEMORY
            # ENGLISH/SPANISH?
            # CODE INTERPRETER

            # llm_chain = LLMChain(llm=llm, prompt=prompt)

            # response = llm_chain.run({"query": query})

            # logger.info(response)

            # vectorstore = Pinecone(index=index, embedding_function=embed.embed_query, text_key='text')
            # retriever = RetrievalQA.from_chain_type(
            #     llm=llm, 
            #     chain_type="stuff",
            #     retriever=vectorstore.as_retriever()
            # )

            # tool_desc = "Use this tool to give me answers based on all the information stored in the Pinecone vector DB. You can also be asked follow up questions based on your answers to my queries."

            # tools = [Tool(name="Carlos' VectorDB", func=retriever.run, description=tool_desc)]

            # memory = ConversationBufferWindowMemory(llm=llm, prompt=prompt, memory_key="chat_history", return_messages=True)

            # agent = initialize_agent(
            #     llm=llm,
            #     tools=tools,
            #     agent="chat-conversational-react-description",
            #     memory=memory,
            #     verbose=True
            # )

            # response = agent.run(query)

            # logger.info(f"Response: {response}")

            # If store is False and namespace did not exist previously, delete the namespace
            if url and not store:
                index.delete(ids=ids)

            return JsonResponse(1, safe=False)

        except Exception as e:
            # Handle any potential exceptions related to document loading and processing
            logger.error(f"Error while analyzing the document: {str(e)}")
            logger.error(traceback.format_exc())  # Log the traceback
            return JsonResponse({'error': 'Error occurred while analyzing the document.'}, status=500)

    return JsonResponse({'error': 'Invalid request method.'})

# Chat Agent view
# def chat_agent(request):
#     return 1
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#         except json.JSONDecodeError:
#             return JsonResponse({'error': 'Invalid JSON data.'}, status=400)

#         input_text = data.get('input')
#         use_serpapi = data.get('use_serpapi', False)  # Retrieve from payload

#         # Create a memory for the conversation
#         memory = ConversationBufferMemory(
#             memory_key="chat_history", return_messages=True)

#         # Initialize the agent
#         tools = []

#         if use_serpapi:

#             def perform_serpapi_search(input_text):
#                 try:
#                     search_results = serpapi.results(input_text)
#                 except Exception as e:
#                     logger.error(
#                         f"Error while fetching search results from SerpAPI: {str(e)}")
#                     return []

#                 formatted_results = []

#                 if "organic_results" in search_results:
#                     organic_results = search_results["organic_results"]

#                     for result in organic_results:
#                         title = result.get("title", "No Title")
#                         link = result.get("link", "#")
#                         snippet = result.get("snippet", "No Snippet")

#                         formatted_result = {
#                             "title": title,
#                             "link": link,
#                             "snippet": snippet
#                         }

#                         formatted_results.append(formatted_result)

#                 return formatted_results

#             logger.info("Using SerpAPI for current search tool.")

#             # Create the SerpAPI tool
#             serpapi_tool = Tool(name="SerpAPI Search Results",
#                                 func=perform_serpapi_search, description="SerpAPI search results")

#             # Append the SerpAPI tool to the tools list
#             tools.append(serpapi_tool)

#         # Initialize the agent with the tools
#         agent_chain = initialize_agent(
#             tools, chat_model, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

#         logger.info(f"Agent chain: {agent_chain}")

#         # Run the agent with user input
#         result = agent_chain.run(input=input_text)

#         logger.info(f"Result from chat agent: {result}")

#         # If the result is a string, try to parse it as JSON
#         if isinstance(result, str):
#             try:
#                 result = json.loads(result)
#             except json.JSONDecodeError:
#                 # If the result is not valid JSON, treat it as a successful response
#                 final_answer = result
#             else:
#                 # If the result is valid JSON, extract the final answer
#                 final_answer = result.get(
#                     "action_input", "I'm sorry, there was an issue with the agent's response.")
#         else:
#             # If the result is already a dictionary, extract the final answer
#             final_answer = result.get(
#                 "action_input", "I'm sorry, there was an issue with the agent's response.")

#         # Return the answer as a JSON response
#         return JsonResponse({'answer': final_answer})

#     return JsonResponse({'error': 'Invalid request method.'})
