import os
import cohere
from dotenv import load_dotenv
from query import query_pinecone

##### Initialize Cohere API client
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

cohere_chat_res_start = co.chat(
    "You are a helpful chatbot that answers questions based on the relevant excerpts provided."
)

conversation_id = cohere_chat_res_start.conversation_id


##### Continue existing chat session
def talk(prompt):
    # results = query_pinecone(prompt)
    # pre_prompt = ""
    # i = 1
    # for match in results["matches"]:
    #     if match["score"] > 0.0:
    #         pre_prompt = (
    #             pre_prompt + str(i) + "." + "\n" + match["metadata"]["text"] + "\n\n"
    #         )
    #         i += 1

    # prompt = "Excerpts: \n" + pre_prompt + "\n\n" + "Query: " + prompt

    response = co.chat(prompt, conversation_id=conversation_id)

    print("\n\n ***************RETRIEVED MATERIAL*************** \n\n " + prompt)
    return response.text


# take prompt from user and call talk function. run this in loop till user exits
while True:
    prompt = input("You: ")
    if prompt == "exit":
        break
    print("Bot: ", talk(prompt))
