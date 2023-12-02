import openai
import os
import requests
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

api_key = os.environ["AZUREOPENAPIKEY"]

service_name = "geicohackcognitivesearch01"
admin_key = "gsdnhH3XQg24VGrdZUOQRU2Avecw9eZobqONAiga1mAzSeCz5nhE"
openai.api_type = "azure"
openai.api_base = "https://geicohackopenai8.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = api_key
index_name = "claimstrainingtranscripts-index"

# Create an SDK client
endpoint = "https://{}.search.windows.net/".format(service_name)

search_client = SearchClient(
    endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(admin_key)
)


def query_cognitive_search(question):
    # search_results = search_client.search(search_text=question, top=1)
    search_results = search_client.search(
        query_type="semantic",
        query_language="en-us",
        semantic_configuration_name="semanticconfig",
        search_text=question,
        select="metadata_storage_path",  # 'content'
        # query_answer='extractive',
        query_caption="extractive",
        top=1,
    )
    return search_results


def generate_reply(question, context):
    prompt = f"You are a GEICO Claims Agent training in the Claim department. Given a question and context, provider accurate answers.\nQuestion: {question}\nContext: {context}\nAnswer:"

    response = openai.Completion.create(
        engine="hackgroup37textdavinci003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=350,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )

    # extract the reply from the openai response
    reply = response.choices[0].text.strip()

    return reply


# Function to process user's question and generate a reply
def process_question(question):
    search_results = query_cognitive_search(question)  # query cognitive search

    if search_results:
        # extract relevant info from search result
        context = extract_relevant_info(search_results)

        reply = generate_reply(question, context)

        return reply

    else:
        return "Couldn't find an answer to your question. Please try again."


def extract_relevant_info(search_results):
    relevant_info = []

    for result in search_results:
        relevant_info.append(result)

    search_captions = "\n".join([c.text for c in relevant_info[0]["@search.captions"]])
    return search_captions


user_question = "What is reserving?"
reply = process_question(user_question)
print(reply)
