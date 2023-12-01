import csv
import openai
import json
import time
from dotenv import load_dotenv
import os
load_dotenv()

data_list = []

# Open the CSV file with the correct encoding
with open('reach.csv', 'r', encoding='utf-8') as f:
    # Create a CSV reader object
    reader = csv.reader(f)

    # Iterate over each row in the CSV file
    for row in reader:
        # Print each column value
        fbid = row[0]
        pagename = row[1]
        review = row[2]
        
        # Create an object and store it in a dictionary
        data_object = {
            'fbid': fbid,
            'pagename': pagename,
            'review': review
        }
        
        # Append the object to the data_list
        data_list.append(data_object)



openai.api_key = os.getenv("OPENAI_KEY")



def toDict(message):
    response = {
        'role': message['role'],
        'content': message['content']
    }
    return response

pages = {
    
}
print('a' in pages)

messages = [
    {"role": "user", "content": "Just respond with a number. Rate by sentiment analysis, 1 - 5"}
]
for data in data_list:
    if data['review'] == 'text':
        continue
    review = "Review: " + data['review']
    instructions = ''''
        Rate the review by sentiment analysis from 1 to 5.
        Do sentiment analysis by category.
        categories are: cleanliness, safety, accessibility, natural beauty, accomodations, recreational activity
        give each score by category

        respond in a json form:
        {
            "cleanliness": score,
            "safety": score,
            "accessibility": score,
            "natural_beauty": score,
            "accommodations": score,
            "recreational_activity": score,
        }
    '''
    message = ({"role": "user", "content": review + instructions})
    print(message['content'])
    completion = None
    retries = 10
    while retries > 0:   
        try: 
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[message], request_timeout=15 * (11-retries))
            rating = completion.choices[0].message['content']
            print(rating)
            dict_data = json.loads(rating)
            dict_data["review"] = data["review"]
            print(dict_data)
            print(completion)

            if data['pagename'] in pages:
                pages[data['pagename']].append(dict_data)
            else:
                pages[data['pagename']] = []
                pages[data['pagename']].append(dict_data)
            break
        except Exception as e:
            retries -= 1
            time.sleep(5)

    print(pages)
    json_data = json.dumps(pages)

    file_path = "data.json"

    with open(file_path, "w") as json_file:
        json_file.write(json_data)




print(pages)
json_data = json.dumps(pages)

file_path = "data.json"

with open(file_path, "w") as json_file:
    json_file.write(json_data)