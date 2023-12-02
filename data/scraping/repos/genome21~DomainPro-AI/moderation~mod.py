from flask import Flask, request, jsonify, render_template
from langchain import PromptTemplate
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import json
import sys
from langchain.chat_models import ChatOpenAI
#import sys.argv

app = Flask(__name__)

chat = ChatOpenAI(temperature=0.7, openai_api_key="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")

# template = """
# You are an accomplished Principal Google Cloud Architect who has worked for Google Cloud since 2014.  Answer questions with a summary then lay out the steps in a list.

# {question}
# """

def testing(output):
        if output == "True":
            print(f"Content was flagged for violating terms of use")
            return output

        if output == "False":
            print(f"Content passed moderation")
            return output

def content_moderation(content):
    #cresponse = openai.ContentFilter.create(
    #prompt=f"{content}"
    cresponse = openai.Moderation.create(
        input=f"{content}"
    )
    output = cresponse["results"][0]["flagged"]
    print(f"Content moderation flag: {output}")
    print("--------------------")
    testing(output)
    #print(f"Content moderation result: {cresponse}")
    return output




print("python script initiated")

question2 = sys.argv[1]
print(f"Received question: {question2}")

content = question2
zresult = content_moderation(content)
print(f"zresult = {zresult}")

print("sending question to openai-1")


# class SetEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, set):
#             return list(obj)
#         return json.JSONEncoder.default(self, obj)

print("received response from openai-1")

if zresult == True:
    #data_str = json.dumps(zresult, cls=SetEncoder).replace("\\n", "<br>")
    print("Content violates terms of use.")
if zresult == False:
    print(zresult)
    print("Content is acceptable.")
    exit()
else:
    exit()
    # print("sending question to openai-2")
    # zresponse = chat([
    #     SystemMessage(content="Answer questions with a short summary then lay out any necessary steps in a list.  Format your responses in markdown."),
    #     HumanMessage(content=question2)
    # ])
    # print("received response from openai-2")
    # print(zresponse)

#data_str = json.dumps(zresponse.content, cls=SetEncoder).replace("\\n", "<br>")

#print(f"data encoded: {data_str}")


# print("-------------------")
# print(f"answer: {data_str}")


