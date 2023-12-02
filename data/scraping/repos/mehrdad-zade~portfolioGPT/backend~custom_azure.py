import openai
from mySecrets import Azure_openai_api_type, Azure_openai_api_version, Azure_openai_api_base, Azure_openai_api_key, Azure_openai_api_deployment_name

def chatGPT3_response(prompt):

    openai.api_type = Azure_openai_api_type
    openai.api_version = Azure_openai_api_version
    openai.api_base = Azure_openai_api_base # your openai api endpoint
    openai.api_key = Azure_openai_api_key # your openai api key
    deployment_name = Azure_openai_api_deployment_name # your azure ai deployment name
    
    res = openai.ChatCompletion.create(
        engine = deployment_name,
        messages = [
                {"role": "user", "content": prompt}]
        )
    return res["choices"][0]["message"]["content"]

# # Example usage
# prompt = "What is the capital of France?"
# response = chatGPT3_response(prompt)
# print(response)


def gpt_res_is_invalid(gptResponse):
    if "september 2021" in gptResponse.lower() or "language model" in gptResponse.lower() or gptResponse == '' or 'cannot provide' in gptResponse:
        return True
    return False

# # Example usage
# result = gpt_res_is_invalid("I am an AI language model and do not have access to the latest financial information. However, you can check the investor relations section of Onex Corporation's website, or financial news sources for information on their Q1 results.")
# if result:
#     print("The similar sentence is present.")
# else:
#     print("The similar sentence is not present.")




################## Experiment - requires RS access on Az

# from azure.ai.language.conversation import ConversationalLanguageClient
# from azure.ai.language.conversation.models import AnalyzeConversationOptions
# from azure.core.credentials import AzureKeyCredential
# from azure.storage.blob import BlobServiceClient
# from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
# from mySecrets import LOCAL_PATH, AZURE_CONVERSATIONAL_LANGUAGE_ENDPOINT, AZURE_CONVERSATIONAL_LANGUAGE_SUBSCRIPTION_KEY, AZURE_BLOB_CONNECTION_STRING, AZURE_BLOB_CONTAINER

# vIdx = LOCAL_PATH + 'data/source_of_knowledge/vectorIndex.json'


# def createVectorIndex():
#     max_input = 4096
#     tokens = 256
#     chunk_size = 600
#     max_chunk_overlap = 20

#     prompt_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

#     # Create Conversational Language client
#     conversational_language_credentials = AzureKeyCredential(AZURE_CONVERSATIONAL_LANGUAGE_SUBSCRIPTION_KEY)
#     conversational_language_client = ConversationalLanguageClient(
#         endpoint=AZURE_CONVERSATIONAL_LANGUAGE_ENDPOINT,
#         credential=conversational_language_credentials
#     )

#     # Load data from Azure Blob Storage
#     blob_service_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
#     container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER)
#     blob_list = container_client.list_blobs()
#     docs = []
#     for blob in blob_list:
#         doc_content = blob.download_blob().readall()
#         docs.append(doc_content.decode('utf-8'))

#     # Define LLM
#     llmPredictor = LLMPredictor(llm=conversational_language_client)  # Replace with Conversational Language client

#     vectorIndex = GPTSimpleVectorIndex(documents=docs, llm_predictor=llmPredictor, prompt_helper=prompt_helper)
#     vectorIndex.save_to_disk(vIdx)

#     return vectorIndex


# def qNa():
#     vIndex = GPTSimpleVectorIndex.load_from_disk(vIdx)
#     while True:
#         prompt = input('Please ask your question here: ')
#         if prompt.lower() != "goodbye.":
#             response = vIndex.query(prompt, response_mode="compact")
#             print(f"Response: {response} \n")
#         else:
#             print("Bot:- Goodbye!")
#             break


# def qNa_source_of_knowledge(question):
#     if os.path.isfile(vIdx):
#         vIndex = GPTSimpleVectorIndex.load_from_disk(vIdx)
#         response = vIndex.query(question, response_mode="compact")
#         return response
#     return "No source of knowledge found. Please upload documents first."


# # Set the values for Azure Conversational Language and Blob Storage
# AZURE_CONVERSATIONAL_LANGUAGE_ENDPOINT = "<your-conversational-language-endpoint>"
# AZURE_CONVERSATIONAL_LANGUAGE_SUBSCRIPTION_KEY = "<your-conversational-language-subscription-key>"
# AZURE_BLOB_CONNECTION_STRING = "<your-blob-connection-string>"
# AZURE_BLOB_CONTAINER = "<your-blob-container>"

# # Uncomment the line below to create the vector index (requires documents to be uploaded to Azure Blob Storage)
# # vectorIndex = createVectorIndex()

# # qNa()
