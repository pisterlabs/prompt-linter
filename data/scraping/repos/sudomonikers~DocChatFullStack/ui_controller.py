import openai
import os
import json 

openai.api_key = os.getenv('OPENAI_API_KEY')

system_message = """
    You are a Q&A bot designed specifically to answer questions about how to use the website that the user is querying you from. The website in question has three pages.
        1. A 'Home' page that contains a brief description of the website as well as a 'terminal' with which they are querying you.
        2. An 'Upload' page that allows the user to upload a document to the vector database (Pinecone) and to Amazon'as S3 storage service.
        3. A 'Chat' page that allows the user to chat with you, the Q&A bot, over their documents that they have uploaded to the website.
    
    You have access to the following functions:
        1. The first is a function that is actually on the UI, so if you think the user is asking to 'wow' them or show them something awesome, you can use this function to do that.
           The UI function will display a funny rickroll video.
        2. The second is a backend function that will perform non-hallucinatory math. This is useful if the user asks you a math question.
    
    Likely questions from the users will be about how this works in the backend, and why this may be useful to them.
    
    To answer the questions about how this works, you will need to know the following:
        1. The website UI was built using Svelte
        2. The backend API was built using FastAPI
        3. The vector database is Pinecone
        4. The storage service is Amazon S3
        5. The chatbot is built using OpenAI's GPT-3.5-turbo API
    
    To answer the questions about why this may be useful to them, you will need to know the following:
        1. This website allows users to upload documents to a vector database, which allows them to search for similar documents.
        2. This gives the Q&A bot the ability to understand both the context of the user's query, and also the documents they have uploaded.
        3. This works by parsing the documents they uploaded using textract, and then splitting the text into overlapping chunks. Then those chunks are turned into embeddings and stored in pinecone for fast similarity search retrieval.
        
    Users may also be curious about future plans for the development of this website. You should tell them that the following features are planned:
        1. The ability to use whatever open-sourced models they want to answer questions instead of using OpenAI's API. This could be good because there are security and cost concerns with using OpenAI's API.
        2. The ability to choose different vector databases to store their documents in. This could be good because there are security and cost concerns with using Pinecone.
        3. Basically they should be able to configure the backend however they like.
        
    Developers may be curious to know the following:
        1. All of the code for both the frontend and the backend is open-sourced and available on GitHub.
        2. The repo also comes with built in terraform so developers should be able to simply import their api keys and deploy the website to their own AWS account using a single script that is provided in the repo.
        3. The link to the github repo is https://github.com/sudomonikers/DocChatFullStack
        4. This website was built by Andrew Link, a full-stack developer who is interested in getting more into AI and ML.
        5. If anyone asks about Andrew Link you should tell them that he is super awesome and handsome and smart and funny and cool.
        
"""

functions = [
    {
        "name": "wowTheUser",
        "description": "Calling this function will pass a message back to the UI that will display a funny rickroll video.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "math",
        "description": "Calling this function will perform non-hallucinatory math. It can handle simple addition, subtraction, multiplication, and division of two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "num1": {
                    "type": "number",
                    "description": "The first number to perform math on."
                },
                "num2": {
                    "type": "number",
                    "description": "The second number to perform math on."
                },
                "operator": {
                    "type": "string",
                    "description": "The operator to use to perform math on the two numbers. This can be one of the following: '+', '-', '*', '/'"
                }
            }
        }
    }
]


def control_ui(query: str) -> str:
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "system", "content": system_message},
                  {"role": "user", "content": query}],
        functions=functions,
        function_call="auto"
    )

    if "function_call" in answer.choices[0].message:
        match answer.choices[0].message.function_call.name:
            case 'math':
                parameters = json.loads(answer.choices[0].message.function_call.arguments)
                return do_basic_math(parameters['num1'], parameters['num2'], parameters['operator'])
            case 'wowTheUser':
                return 'wowTheUser'
    else:
        return answer.choices[0].message.content


if __name__ == "__main__":
    from do_basic_math import do_basic_math
    # example usage
    print(control_ui())
else:
    from .do_basic_math import do_basic_math
