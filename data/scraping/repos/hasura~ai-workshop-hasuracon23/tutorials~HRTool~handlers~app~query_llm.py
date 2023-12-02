from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# GraphQL endpoint URL
GRAPHQL_ENDPOINT = "http://graphql-engine:8080/v1/graphql"


def get_prompt(request):
    user_query = request['input']['user_query']

    # Add authenticated session variables as headers along with the admin secret
    gql_headers = request['session_variables']
    gql_headers['x-hasura-admin-secret'] = 'secret'

    # Create a GraphQL client with the request transport
    transport = RequestsHTTPTransport(
        url=GRAPHQL_ENDPOINT, headers=gql_headers)
    client = Client(transport=transport)

    # Send the GraphQL request
    gql_query = gql("""
            query getItems($user_query: text!) {
                Resume(where: { vector: { near_text: $user_query}}, limit: 3) {
                    content
                    application_id
                }
            }
        """)
    result = client.execute(gql_query, variable_values={
                            'user_query': user_query})
    # resumes = result['data']['Resume']
    resumes = result["Resume"]

    prompt = """
    You are a helpful Question Answering bot. 
    You are provided with content from a few resumes and a question.
    Answer the question based on the content of the resumes.
    Provide your reasoning.

    Question: {question}"""
    prompt += user_query

    for resume in resumes:
        prompt += "Resume:"
        prompt += resume["content"]
        prompt += "with Application ID: "
        prompt += resume["application_id"]
        prompt += "\n"

    return prompt


def query_llm(request, headers):
    llm = OpenAI(model="text-davinci-003",
                 openai_api_key=os.environ['OPENAI_APIKEY'])
    prompt = get_prompt(request)
    chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt))
    return str(chain.run(
        {"question":request["input"]["user_query"]}
        ))
