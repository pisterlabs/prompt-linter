# You need to perform these commands first in the top directory of the repository
# curl https://d2eo22ngex1n9g.cloudfront.net/Documentation/SDK/bedrock-python-sdk.zip --output bedrock-python-sdk.zip
# unzip -o bedrock-python-sdk.zip
# pip install -r requirements_bedrock.txt
# pip install *.whl
# mkdir texts
# curl https://raw.githubusercontent.com/synedra/astra_vector_examples/main/romeo_astra.json --output texts/romeo_astra.json

# Use astra cli to create .env file then add AWS keys to it
# .env file should have the following:
#   - AWS_ACCESS_KEY_ID
#   - AWS_SECRET_ACCESS_KEY
#   - AWS_SESSION_TOKEN
#   - ASTRA_DB_KEYSPACE
#   - ASTRA_DB_APPLICATION_TOKEN

from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import boto3, json, os, sys
from getpass import getpass
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

cluster = Cluster(
    cloud={
        "secure_connect_bundle": os.environ["ASTRA_DB_SECURE_BUNDLE_PATH"],
    },
    auth_provider=PlainTextAuthProvider(
        "token",
        os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    ),
)
session = cluster.connect()

bedrock = boto3.client(
    "bedrock",
    "us-west-2",
    endpoint_url="https://invoke-bedrock.us-west-2.amazonaws.com",
)
br_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-g1-text-02", client=bedrock
)

myCassandraVStore = Cassandra(
    embedding=br_embeddings,
    session=session,
    keyspace=os.environ["ASTRA_DB_KEYSPACE"],
    table_name="shakespeare_act5",
)
quote_array = json.load(open("texts/romeo_astra.json"))
for index in range(len(quote_array)):
    location = ""
    if quote_array[index]["ActSceneLine"] != "":
        (act, scene, line) = quote_array[index]["ActSceneLine"].split(".")
        location = "Act {}, Scene {}, Line {}".format(act, scene, line)
    quote_input = "{} : {} : {}".format(
        location, quote_array[index]["Player"], quote_array[index]["PlayerLine"]
    )
    input_document = Document(page_content=quote_input)
    print(quote_input)
    myCassandraVStore.add_documents(documents=[input_document])

# Enter a question about Romeo and Astra (Like 'How did Astra die?')
QUESTION_FOR_MODEL = "How did Astra Die?"

generation_prompt_template = """Please answer a question from a user.
      Create a summary of the information between ## to answer the question.
      Your task is to answer the question using only the summary using 20 words

      #
      {context}
      #

      question= {question}
      Answer: """

retriever = myCassandraVStore.as_retriever(
    search_kwargs={
        "k": 2,
    }
)
output = retriever.get_relevant_documents(QUESTION_FOR_MODEL)
prompt = PromptTemplate.from_template("{page_content}")
context = ""
for document in output:
    context += " *** " + document.page_content

print(context)
llm_prompt = generation_prompt_template.format(
    question=QUESTION_FOR_MODEL,
    context=context,
)

body = json.dumps({"inputText": llm_prompt})
modelId = "amazon.titan-tg1-large"
accept = "application/json"
contentType = "application/json"
response = bedrock.invoke_model(
    body=body, modelId=modelId, accept=accept, contentType=contentType
)
response_body = json.loads(response.get("body").read())
print("Question: " + QUESTION_FOR_MODEL)
print("Answer: " + response_body.get("results")[0].get("outputText"))
