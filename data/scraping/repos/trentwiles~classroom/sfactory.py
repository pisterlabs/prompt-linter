import os
import classroom
import json
import openai
from dotenv import load_dotenv
from weasyprint import HTML, CSS
import datetime
import time

load_dotenv()

def chatGPT(message):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": message}
    ]
    )

    return completion.choices[0].message["content"]

def determineWorkTime(dueIn, difCoef, totalClasses):
    print(difCoef)
    # This is the procrastination code
    # dueIn is going to be an int

    if dueIn >= 0.00 and dueIn < 1.00:
        return (difCoef * 4)
    elif dueIn >= 1.00 and dueIn < 24.00:
        return (difCoef * 3)
    elif dueIn >= 24.00 and dueIn < 48.00:
        return (difCoef * 2)
    elif dueIn >= 48.00 and dueIn < 96.00:
        return (difCoef * 1)
    else:
        return (difCoef * 0.5)

def createS(secure_id):
    burnerVariable = ""

    # "difCoef" is a numeric value to determine how hard each class is in comparison to others
    difCoef = 1 # starts at 1
    toSubtractEachTime = 0 # by default
    totalClasses = 0
    workToDoOrdered = [] # this list holds all of the work in order, as designated by ChatGPT and the due time
    # Step One: Set up the chatGPT prompt
    chatGPTprompt = "Assuming that math typically has the most homework, next science/physics, then english, then foreign languages, then electives, how would this list of class titles rank? Please list them in order, one per line, no other text. Thanks! Here is the list: \n"
    # Step Two: Check if the configuration files set up before the dashboard is accessed exist
    if not os.path.isfile('temp/classes-' + str(secure_id) + '.json') or not os.path.isfile('temp/settings-' + str(secure_id) + '.json'):
        return Exception(FileNotFoundError)
    with open('temp/classes-' + str(secure_id) + '.json') as r:
        s = json.loads(r.read())
        totalClasses = len(s)
        print(totalClasses)
        toSubtractEachTime = 1/totalClasses
    # Step Three: Open the classes file. This contains all of the class IDs
    with open('temp/classes-' + str(secure_id) + '.json') as r:
        s = json.loads(r.read())

        for x in s:
            if x != None:
                n = classroom.getClassByID(int(x), secure_id)["name"]
                # Step Four: add each class name to the ChatGPT prompt so they can be ordered
                chatGPTprompt += n + "\n"
    print(chatGPTprompt)
    #time.sleep(60)
    response = chatGPT(chatGPTprompt)
    for orderedClass in response.split("\n"):
        print("30 minutes for " + orderedClass)
        print(classroom.getCourseLoadByName(orderedClass, secure_id))
        try:
            for x in classroom.getCourseLoadByName(orderedClass, secure_id)["courseWork"]:
                # Step Five: Now fetch all of the course work for each class
                #print(x)
                try:
                    dueAt = int(datetime.datetime(x["dueDate"]["year"], x["dueDate"]["month"], x["dueDate"]["day"]).timestamp())
                    timeNow = int(time.time())
                    dueIn = (dueAt - timeNow) / (3600 * 24)
                    # if the dueDate minus the current time is larger than 0, the assignment is due
                    if dueIn > 0:
                        #print(x[""])
                        #print("Due in " +  + " days")
                        #print("Due @ timestamp: " + str(dueAt))
                        #print(x)
                        hoursUntilDue = round(dueIn, 2)
                        allocatedTime = determineWorkTime(hoursUntilDue, difCoef, totalClasses)
                        workToDoOrdered.append({"title": x["title"], "description": x["description"], "className": orderedClass, "dueIn": str(hoursUntilDue), "allocatedTime": round(allocatedTime, 2)})
                    else:
                        burnerVariable = ""
                except:
                    #print(" **** NO DUE DATE ***** ")
                    burnerVariable = ""
        except:
            print("no work to do for this class")
        difCoef -= toSubtractEachTime
    return workToDoOrdered

def assemble(tasks, secure_id):
    css = "<style>body{ font-family: 'arial'; }</style>"
    html = "<html><body>" + css + "<h1>Tasks for Today</h1><table><tr><th>Assignment</th><th>Class</th><th>Time</th></tr>"
    # "tasks" is a list of items with allocated time
    for x in tasks:
        html += "<tr><td>" + x["title"] + "</td><td>" + x["className"] + "</td><td>" + str(round(float(x["allocatedTime"]) * 60)) + " minutes</td></tr>"
    print("=======================")
    print(x["allocatedTime"])
    print(float(x["allocatedTime"]))
    print(float(x["allocatedTime"]) * 60)
    html += "</table></body></html>"
    HTML(string=html).write_pdf('output-' + str(secure_id) + '.pdf')
    return None