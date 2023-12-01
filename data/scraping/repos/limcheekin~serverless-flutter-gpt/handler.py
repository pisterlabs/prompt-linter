import json
import os
import base64
import urllib.parse
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient

CONVERSATIONAL_MEMORY_NOT_IMPLEMENTED="\nBy the way, I have a memory of a goldfish as the conversational memory is not implemented, so I don’t remember your previous questions or my answers. Please give me some context information when you ask me something, or I’ll just make up random stuff. :)"

system_template="""Use the following pieces of context to answer the users question. 
If you don't know the answer, just say "Hmm..., I'm not sure.", don't try to make up an answer.
ALWAYS return a "Sources" part in your answer.
The "Sources" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo

Sources:
1. abc
2. xyz
```

Begin!
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

url = os.environ.get("QDRANT_URL")
api_key = os.environ.get("QDRANT_API_KEY")
qdrant = Qdrant(QdrantClient(url=url, api_key=api_key), "docs_flutter_dev", embedding_function=OpenAIEmbeddings().embed_query)

def get_chain(vectorstore, prompt):
    chain_type_kwargs = {"prompt": prompt}
    chain = VectorDBQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), 
        chain_type="stuff", 
        vectorstore=vectorstore,
        chain_type_kwargs=chain_type_kwargs,
        reduce_k_below_max_tokens=True
    )
    return chain

def ask_question(event, context):

    message = "The chatbot seems not working!\nPlease report the issue/error at https://github.com/limcheekin/serverless-flutter-gpt/issues.\nThanks."

    print("event.get('source')", event.get("source"))

    if event.get("source") == "serverless-plugin-warmup":
        message = 'WarmUP - Keep the Lambda warm!'

    else: 
        print(json.dumps(event))
        body = event.get("body")
        if event.get("isBase64Encoded"): 
            base64_bytes = base64.b64decode(body)
            body = base64_bytes.decode("utf-8")
            body = urllib.parse.parse_qs(body)
        context = json.loads(body['contextobj'][0])    
        senderobj = json.loads(body["senderobj"][0])
        messageobj = json.loads(body['messageobj'][0])

        qa_chain = get_chain(qdrant, prompt)
        question = messageobj['text']
        if question == "/start":
            message = f"Hello {senderobj['display']}, thanks for connecting. I'm {context['botname']}, you can type /help to get to know me.\nHow can I help you today?"
        elif question == "/help":
            message = "It look like you need some help, type one of the following commands:\n/start to start asking questions,\n/about to find out more about me and my creator,\n/feedback to feedback or report issue/error, or\n/help to see list of commands."
        elif question == "/about":
            message = f"I'm {context['botname']}, an AI chatbot that can answer questions about Flutter.\nMy knowledge is derived from https://docs.flutter.dev/, cutoff date of March 2023 and created by @limcheekin, feel free to contact him at https://www.linkedin.com/in/limcheekin/.{CONVERSATIONAL_MEMORY_NOT_IMPLEMENTED}"
        elif question == "/feedback":
            message = "Thanks for your feedback. Please share your idea or report any issue/error at https://github.com/limcheekin/serverless-flutter-gpt/issues."
        else:
            result = qa_chain({"question": question})
            message = result['answer']
            message = message.replace("./site", "https://docs.flutter.dev")
            print("\n\n")
            print(result)

    print(message)

    response = {
        "statusCode": 200,
        "body": message,
        "headers": {
            "Content-Type": 'application/json'
        }
    }

    return response

def test_ask_question(event_file):
    with open(event_file) as json_file:
        file_contents = json_file.read()
    event = json.loads(file_contents)
    response = ask_question(event, None)
    print(f"response:\n{response}")
  
if __name__ == "__main__":
    test_ask_question("test/data/event_start.json")
    test_ask_question("test/data/event_help.json")
    test_ask_question("test/data/event_about.json")
    test_ask_question("test/data/event_question.json")   
    test_ask_question("test/data/event_openai_cutoff.json")