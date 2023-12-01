import csv
import openai
import json
from dotenv import load_dotenv
import os
import time
load_dotenv()

data_list = []

with open('reach.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        fbid = row[0]
        pagename = row[1]
        review = row[2]
        data_object = {
            'fbid': fbid,
            'pagename': pagename,
            'review': review
        }
        data_list.append(data_object)


openai.api_key = os.getenv("OPENAI_KEY")

def toDict(message):
    response = {
        'role': message['role'],
        'content': message['content']
    }
    return response

pages = {}
messages = [
    {"role": "user", "content": "Just respond with a number. Rate by sentiment analysis, 1 - 5"}
]

for data in data_list:
    if data['review'] == 'text':
        continue
    keywords = "These are the keywords to choose from:  [rooms, cottage, villa, tent, camping, pools] \n"
    review = "Review: " + data['review']
    instructions = ''''
    
Analyze the review, which of the keywords provided above applies to the review? only select one keyword and if there is none then only output empty
respond in only one word,
do not add explanations
    '''
    message = ({"role": "user", "content": keywords + review + instructions})

    retries = 3
    while retries > 0:   
        try: 
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[message], request_timeout=15 * (11-retries))
            keywords = completion.choices[0].message['content']
            keywords_list = list(set(keywords.split(', ')))
            if data['pagename'] in pages:
                pages[data['pagename']] = list(set(pages[data['pagename']] + keywords_list))
            else:
                pages[data['pagename']] = []
                pages[data['pagename']].append(keywords)
            print(keywords_list)
            break
        except Exception as e:
            retries -= 1
            time.sleep(5)

    json_data = json.dumps(pages)
    file_path = "accomodation.json"
    with open(file_path, "w") as json_file:
        json_file.write(json_data)




print(pages)
json_data = json.dumps(pages)
file_path = "accomodation_final.json"

with open(file_path, "w") as json_file:
    json_file.write(json_data)