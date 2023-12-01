from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List,Optional
import json
import urllib.parse
import urllib.request
import requests
import os
from openai import OpenAI


app = FastAPI()
import requests
responseSummer = requests.get("https://api.peterportal.org/rest/v0/schedule/soc?term=2023%20Summer10wk&department=COMPSCI")
responseFall = requests.get("https://api.peterportal.org/rest/v0/schedule/soc?term=2023%20Fall&department=COMPSCI")
responseWinter = requests.get("https://api.peterportal.org/rest/v0/schedule/soc?term=2023%20Winter&department=COMPSCI")
responseSpring = requests.get("https://api.peterportal.org/rest/v0/schedule/soc?term=2023%20Spring&department=COMPSCI")
courses = requests.get("https://api.peterportal.org/rest/v0/courses/all").json()
summer = responseSummer.json()
fall = responseFall.json()
winter = responseWinter.json()
spring = responseSpring.json()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FormData(BaseModel):
    term1: str
    year1: str
    units: str
    term2: str
    year2: str
    minor: str
    text_area: List[Optional[List[Optional[str]]]] = [ [] ]  # Initialize with an empty list of Optional[str]
    specialization: str
    summertogle: str

lowerdiv = ["I&C SCI 31","I&C SCI 6B","MATH 2A","MATH 2B","I&C SCI 32","I&C SCI 6N","I&C SCI 6D","I&C SCI 33","IN4MATIX 43", "I&C SCI 6N","STATS 67", "I&C SCI 45C", "I&C SCI 46" , "I&C SCI 51", "I&C SCI 53"]
upperdivA = ["COMPSCI 161","I&C SCI 139W"]
upperdivB = [
    "COMPSCI 103", "COMPSCI 104", "COMPSCI 105", "COMPSCI 106", "COMPSCI 107",
    "COMPSCI 108", "COMPSCI 109", "COMPSCI 110", "COMPSCI 111", "COMPSCI 112",
    "COMPSCI 113", "COMPSCI 114", "COMPSCI 117", "COMPSCI 118", "COMPSCI 122B",
    "COMPSCI 122C", "COMPSCI 122D", "COMPSCI 125", "COMPSCI 133", "COMPSCI 142B",
    "COMPSCI 143B", "COMPSCI 145", "COMPSCI 147", "COMPSCI 153", "COMPSCI 154",
    "COMPSCI 165", "COMPSCI 175", "COMPSCI 180A", "COMPSCI 180B", "COMPSCI 189",
    "IN4MATX 102", "IN4MATX 113", "IN4MATX 115", "IN4MATX 117", "IN4MATX 121",
    "IN4MATX 122", "IN4MATX 124", "IN4MATX 131", "IN4MATX 133", "IN4MATX 134",
    "I&C SCI 162"
]
Algorithms = ["COMPSCI 162", "COMPSCI 163", "COMPSCI 164", "COMPSCI 165", "COMPSCI 166", "COMPSCI 167", "COMPSCI 169"]
Architecture = ["COMPSCI 145", "COMPSCI 147", "COMPSCI 151", "COMPSCI 152","COMPSCI 153","COMPSCI 154"]
BioInformatics = []
@app.post("/course-management")
async def course_management(data: FormData):
    # Access the form data  
    print(data)
    finalList = data.text_area
    for index in range(20):
        if len(finalList) - index <= 0:
            finalList.append([])
        if finalList[index] == None:
            finalList[index] = []

    taken = set([])
    for index, term in enumerate(finalList):
        if index == len(finalList)-1:
            break
        for element in term:
            taken.add(element)
            for classes in courses:
                if element.replace(" ", "") == classes["id"]:
                    for prereqs in classes["prerequisite_for"]:
                        if prereqs in lowerdiv and prereqs not in taken:
                            finalList[index+1].append(prereqs)
                            taken.add(prereqs)
                    
    for index in range(20):
        for i in range(5):
            if len(finalList[index]) < 5:
                 finalList[index].append('')

    #finalList = get_course_list(data.specialization)
    print(finalList)
    return {"message": finalList}
'''
    os.environ['OPENAI_API_KEY'] = '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' 
print(os.environ['OPENAI_API_KEY'])

course_list = ['COMPSCI103', 'COMPSCI111', 'COMPSCI112', 'COMPSCI113', 'COMPSCI114', 'COMPSCI115', 'COMPSCI116', 'COMPSCI117', 'COMPSCI118',
               'COMPSCI121', 'COMPSCI122A', 'COMPSCI122B', 'COMPSCI122C', 'COMPSCI122D', 'COMPSCI125', 'COMPSCI131', 'COMPSCI132', 'COMPSCI133',
               'COMPSCI134', 'COMPSCI137', 'COMPSCI141', 'COMPSCI142A', 'COMPSCI142B', 'COMPSCI143A', 'COMPSCI143B', 'COMPSCI145', 'COMPSCI145L',
               'COMPSCI146', 'COMPSCI147', 'COMPSCI151', 'COMPSCI152', 'COMPSCI153', 'COMPSCI154', 'COMPSCI161', 'COMPSCI162', 'COMPSCI163', 'COMPSCI164',
               'COMPSCI165', 'COMPSCI166', 'COMPSCI166', 'COMPSCI167', 'COMPSCI169', 'COMPSCI171', 'COMPSCI172B', 'COMPSCI172C', 'COMPSCI175', 'COMPSCI177',
               'COMPSCI178', 'COMPSCI179', 'COMPSCI180A', 'COMPSCI180B', 'COMPSCI183', 'COMPSCI184A', 'COMPSCI184C', 'COMPSCI190', 'COMPSCI199',
            "IN4MATX 102", "IN4MATX 113", "IN4MATX 115", "IN4MATX 117", "IN4MATX 121", "IN4MATX 122", "IN4MATX 124", "IN4MATX 131", "IN4MATX 133", "IN4MATX 134"]



def chatgpt_prompt(lst: str, specialization):
    client = OpenAI()

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a personal assistant, skilled in giving a personalized schedule of classes to college students. Choose the 10 best classes that will help studencts with a computer science degree"},
        {"role": "user", "content": 'Here are the prompts: ' + lst + 'please give only the IDs of the 10 best classes related to this specialization in a python list with the ids as strings: ' + specialization}
      ]
    )

    return completion.choices[0].message.content


def get_course_data(course):
    url = 'https://api.peterportal.org/rest/v0/courses/' + course
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(request)
    json_text = response.read()
    json_file = json.loads(json_text)
    return [json_file['id'], json_file['description']]


# Call get course List to get the 10 best classes to take
def get_course_list(specialization):
    new_list = []
    for item in course_list:
        lst = get_course_data(item)
        new_list.append(lst)

    item = chatgpt_prompt(str(new_list), specialization)

    counter = 0
    for i in range(20):
        for j in range(5):
            second_counter = 0
            if finalList[i][j] !='':
                pass
            else:
                if counter < 10 and second_counter <2:
                    finalList[i][j] = item[counter]
                    counter += 1
                    second_counter += 1

    return finalList
'''
