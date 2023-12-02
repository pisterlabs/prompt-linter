import pandas as pd
from test_mulit import run_multi_client
from openai_multi_client import OpenAIMultiClient, Payload
# Remember to set the OPENAI_API_KEY environment variable to your API key
import openai
openai.api_key = "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"
system_message = "Du skal skrive en bakgrunns historie for en karakter målet er å lage karakterer som er så virkelige som mulig."
topic = " Lag en bakgruns historie for en karakter som er mellom 8-12 som er beskrevet med attributer i denne listen"
 

def get_attributes_list(path):
    # reads xlsx file
    df = pd.read_excel(path)
    attributes_list = []
    # df to list
    for index, row in df.iterrows():
        attributes = ""
        for column in df.columns:
            attributes += f"[ {column} {row[column]} ]"
        attributes_list.append(attributes)
    return attributes_list


def history_list_creator(atributes_list):
    history_list = []
    for attributes in atributes_list:
        history = [{"role": "system", "content": system_message}]
        history.append({"role": "assistant", "content" : topic + attributes})
        history_list.append(history)
    return history_list

# data_path = "SyntheticPersons\data\synthpersons.csv"
data_path = "SyntheticPersons\data\Segmentdata_test.xlsx"

attributes_list = get_attributes_list(data_path)
history_list = history_list_creator(attributes_list)
print(history_list[0])
print(len(history_list))


def on_result(result: Payload):

    response = result.response['choices'][0]['message']['content']
       # save responses in a list
    path_to_save = "SyntheticPersons\\data\\responses.txt"
    # create file if it does not exist
    with open(path_to_save, "a") as file:
        file.write( "####" + response + "\n")
    print(response)

def make_requests(history_list, api):
    for history in history_list:
        api.request(data={ "messages": history}, callback=on_result)


api = OpenAIMultiClient(endpoint="chats", data_template={"model": "gpt-3.5-turbo"}, max_retries=1)
api.run_request_function(make_requests(history_list, api))
api.pull_all()
 
