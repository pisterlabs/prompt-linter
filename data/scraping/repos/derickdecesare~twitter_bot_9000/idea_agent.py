import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from pyairtable import Table
import time

load_dotenv()

airtable_api_key = os.environ["AIRTABLE_API_KEY"]
base_id = os.environ["AIRTABLE_BASE_ID"]
table_name = os.environ["AIRTABLE_TABLE_NAME"]
open_ai_api_key = os.environ["OPENAI_API_KEY"]

## THis agent is going to to receive an objective (topic ) and then generate an idea for the research agent (that we have not already created) to research that wil be a good twitter thread
# THen it will store that idea in airtable so we can reference it later

# So first it will pull all the ideas from airtable and then generate a new unique idea, store it in an airtable, and then return it from the function that will be used in another file


table = Table(airtable_api_key, base_id, table_name)


# tables = table.all()

# print(tables)






# # Wait for 10 seconds
# time.sleep(5)

# # Add Research
# research_data = {
#     "Research": "This is the long form research asycn response",
# }
# response_research = table.update(idea_id, research_data)

# # Wait for another 10 seconds
# time.sleep(5)

# # Add Thread
# thread_data = {
#     "Thread": "This is the thread which is also longform async response",
# }
# response_thread = table.update(idea_id, thread_data)








def generate_idea(topic):

    print("Getting ideas from airtable...")
    # Get ideas from airtable
    records = table.all()

    existing_ideas = [record['fields']['Ideas'] for record in records if 'Ideas' in record['fields']]

    print(existing_ideas)

    print("Generating idea...")

    airtable_id = None
    response_content = None

    llm = ChatOpenAI(temperature=0, model_name='gpt-4')

    messages = [
    SystemMessage(content="You are an expert writer that generates one unique, high quality idea for a Twitter thread that will be engaging. Your idea is not politcal or dealing with ethics."),
    HumanMessage(content=f"Here are the list of previous ideas you have generated {existing_ideas}. \n\nPlease generate a new, unique, high quality twitter thread idea on the topic of {topic}. The idea should be short and succinct, ideally in less than ten words. Please format without any punctuation or quotation marks.")
    ]

    response = llm(messages)
    response_content = response.content

    # Add idea to airtable
    print("Adding idea to airtable...")
    idea_data = {
        "Ideas": response_content,
    }
    response_idea = table.create(idea_data)
    airtable_id = response_idea['id']

    return {"airtable_id": airtable_id ,"idea":response_content}


# idea = generate_idea("AI startups")

# print(idea)


