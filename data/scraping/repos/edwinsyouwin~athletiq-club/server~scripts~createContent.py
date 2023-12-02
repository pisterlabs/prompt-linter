# from gpt4all import GPT4All
from langchain.llms import OpenAI, GPT4All
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import CommaSeparatedListOutputParser, ResponseSchema, StructuredOutputParser
import csv
import pandas as pd
from dotenv import load_dotenv
import os
# from pydantic import BaseModel, Field, validator
import json
import logging

load_dotenv(dotenv_path="../../.env")

print(os.getenv("OPENAI_API_KEY"))
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# model = OpenAI(os.getenv("OPENAI_API_KEY"))
llm = OpenAI(model="text-davinci-003" , temperature=0.0)
# llm = GPT4All(model="orca-mini-3b.ggmlv3.q4_0.bin", n_threads=8)


output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()

def generateTopicList(topic, location, list_length=10):
    list_length = str(list_length)

    prompt = PromptTemplate(
            input_variables=["topic", "location", "list_length"],
            template="List the top {list_length} most popular {topic} in the {location}?.\n{format_instructions}",
            partial_variables={"format_instructions": format_instructions}

        )
    chain = LLMChain(llm=llm, prompt=prompt)

    topic_list = output_parser.parse(
    chain.run({
        'topic': topic,
        'location': location,
        'list_length': list_length
        }))


    
    topic_df = pd.DataFrame({topic: topic_list})
    topic_df.to_csv(f"{topic}_{location}_list.csv", index=True, header=True, index_label="Index")

    return topic_df

# class Influencer(BaseModel):
#     name: str = Field(description="Influencer's name")
    # first_name: str = Field(description="Influencer's first name")
    # last_name: str = Field(description="Influencer's last name")
    # topic: str = Field(description="Influencer's topic")
    # location: str = Field(description="Influencer's location")


    # # You can add custom validation logic easily with Pydantic.
    # @validator("setup")
    # def question_ends_with_question_mark(cls, field):
    #     if field[-1] != "?":
    #         raise ValueError("Badly formed question!")
    #     return field





response_schema_dict = {
    "influencers": [
        ResponseSchema(name="Name", description=f"Name of the influencer"),
        ResponseSchema(name="Instagram Handle", description="The influencer's Instagram handle"),        
        ResponseSchema(name="Estimated Followers", description="Estimated number of followers on social media"),
    ],
    "equipment": [
        ResponseSchema(name="Name", description=f"Name of the equipment"),
        ResponseSchema(name="Necessary", description="Is this equipment absolutely necessary to play the sport?"),
        ResponseSchema(name="Average Cost", description="Estimated average cost of the equipment"),
        ResponseSchema(name="Top Brands", description="Top brands for this equipment"),
        ResponseSchema(name="Beginner", description="What should a beginner participant look for?"),
        # ResponseSchema(name="Intermediate", description="What should an intermediate participant look for?"),
        ResponseSchema(name="Advanced", description="What should an advanced participant look for?"),
    ],
    
}





def generateTopicContentDF(noun, topic_item, topic, year, list_length=10, existing_list=None):



    response_schemas = response_schema_dict[noun]


    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    list_length = str(list_length)


    if existing_list:
        prompt = PromptTemplate(
                input_variables=["noun", "topic_item", "year", "list_length"],
                template="""List the top {list_length} most popular {noun} in {topic_item} in {year}.
                
                Your response should absolutely NOT include anyone in this list: """ + existing_list + """

                {format_instructions}
                Wrap your final output with closed and open brackets (a list of json objects) and separate each object with a comma.
                

                """,
                partial_variables={"format_instructions": format_instructions},

            )
    else:
        prompt = PromptTemplate(
                input_variables=["noun", "topic_item", "year", "list_length"],
                template="""List the top {list_length} most popular {noun} in {topic_item} in {year}.
                
                {format_instructions}
                Wrap your final output with closed and open brackets (a list of json objects) and separate each object with a comma.
                
                """,
                partial_variables={"format_instructions": format_instructions},

            )
    

    llm = OpenAI(temperature=0.0)

    chain = LLMChain(llm=llm, prompt=prompt)

    raw_content = chain.run({
        'noun': noun,
        'topic_item': topic_item,
        'year': year,
        'list_length': list_length
        })
    
    # print(influencer)

    if "```json" in raw_content:
        raw_content = raw_content.split("```json")[1].strip()
    if "```" in raw_content:
        raw_content = raw_content.split("```")[0].strip()


    content_json = json.loads(raw_content)

    content_df = pd.DataFrame(content_json)
    

    if existing_list:
        content_df.to_csv(f"scriptfiles/{topic}/{noun}/{topic_item}_{noun}_content.csv", mode='a', header=False, index=False)
    else:
        content_df.to_csv(f"scriptfiles/{topic}/{noun}/{topic_item}_{noun}_content.csv", header=True, index=False)
    # # output_parser.parse(influencer)
    

def getTopicDF(topic, location, list_length=10):
    try:
        topic_list = pd.read_csv(f"scriptfiles/{topic}_list.csv", index_col="Index")
        return topic_list
    except:
        FileExistsError
        # topic_df = generateTopicDF(topic, location, list_length)
        # return topic_df



# def getInfluencerDF(topic, noun):
#     try:
#         influencer_df = pd.read_csv(f"{topic}_{noun}_content.csv", index_col="Index", dropna=True)
#         return influencer_df
#     except:
#         FileExistsError
#         influencer_df = generateTopicContentDF(noun, topic, "2023", 10)
#         return influencer_df
    
generateTopicDF("sports", "United States", 10)
topic_list = getTopicDF("sports", "United States", 11)

print(topic_list)



# generateTopicContentDF("influencers", "American Football", "2023", 5)

# def generateTopicContent(noun, topic, number_of_interations=1):
#     topic_df = getTopicDF(topic, "United States", 11)

#     for i, rows in topic_df.iterrows():
#         topic_item = rows[topic]
#         for i in range(number_of_interations):
#             try:
#                 topic_item_noun_df = pd.read_csv(f"scriptfiles/{topic}/{noun}/{topic_item}_{noun}_content.csv")
#                 existing_topic_item_list = str(topic_item_noun_df["Name"].tolist())
#                 generateTopicContentDF(noun, topic_item, topic, "2023", 1, existing_list=existing_topic_item_list)
#             except Exception as e:
#                 logging.error(str(e) + " " + topic_item)
#                 generateTopicContentDF(noun, topic_item, topic, "2023", 1)
#                 continue


# generateTopicContent("equipment", "sports", 2)