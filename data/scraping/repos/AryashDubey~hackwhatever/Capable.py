import openai
from flask import Flask, request, jsonify

app = Flask(__name__)

API_KEY = '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' 
openai.api_key = API_KEY

def initialize_conversation(data):
    name = data["name"]
    age = data["age"]
    location = data["location"]
    disability = data["disability"]
    gender = data["gender"]

    return [
        {
            "role": "system",
            "content": '''
                This program is supposed to be inclusive of people's disabilities and promote their independence/abilities.
                Be extremely mindful and careful of the language used!
            '''
        },
        {
            "role": "assistant",
            "content": name+" who is a "+gender+" aged "+age+" years old and is situated in " + location + " is " + disability +
                        '''
                Nothing about the disability should be included/mentioned in the response.
                Be very specific in your response with all details (including examples for each suggestion) covered in their entirety.
                Ensure that the suggestions to ''' + name + ''' suit both the location they are from and their disability as well.
            '''
        }
    ]

def get_openai_response(messages):
    query = input("Query:")
    messages.append({
        "role": "user",
        "content": query + " Provide suggestions that are more suitable for the user's specific needs,location, and age, without explicitly mentioning the provided disability(s)!"
    })

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    data = {
        "content":response.choices[0].message.content
    }

    api_call = requests.post(url, json = data)
    
    messages.append(response.choices[0].message)
    return jsonify(response.choices[0].message), 200
    
'''
def main():
    messages = initialize_conversation()
    while True:
       get_openai_response(messages)
'''

@app.route('/run_app',methods=['GET','POST'])
def run_app():
    info = request.json
    conv_history = initialize_conversation(info)
    while True:
        get_openai_response(conv_history)
    
if __name__ == '__main__':
    app.run(host="0.0.0.0",port="5000")

"""
import openai

API_KEY = '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' 
openai.api_key = API_KEY
messages = []

def begin():
    name = input("Enter name:")
    age = input("Enter age:")
    location = input("Enter location:")
    disability = input("Enter limitation:")
    comments = input("Enter other comments:")
    query = input("Query:")
    messages = [
        {
        "role": "system",
        "content": '''
            This program is supposed to be inclusive of people's disabilities and promote their independence/abilities.
            Be extremely mindful and careful of the language used!
        '''
        },
        {
        "role": "assistant",
        "content": name + " who is " + age + " years old and is situated in " + location + " is " + disability +
                   '''
            Nothing about the disability should be included/mentioned in the response.
            Be very specific in your response with all details (including examples for each suggestion) covered in their entirety.
            Ensure that the suggestions to '''+name+''' suit both the location they are from and their disability as well.
        '''
        },
        {
        "role": "user",
        "content": query + " Provide suggestions that are more suitable for someone with specific needs, without explicitly mentioning any disabilities."
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    messages.append(response.choices[0].message)
    print(response.choices[0].message.content)

def repeat():
    query = input("Enter Follow Up Query:")
    messages.append(
        {
        "role":"user",
        "content":query+" Make sure that this is still "+disability+" friendly without explicitly mentioning it."
        }
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    messages.append(response.choices[0].message)
    print(response.choices[0].message.content)

begin()
while True:
    repeat()


messages = [
    {
        "role":"system","content":
        '''
            This program is supposed to be inclusive of people's disabilities and promote their independence/abilities.
            Be extremely mindful and careful of the language used!
        '''
    },
    {
        "role":"assistant","content":
        name+" who is "+age+" years old and is situated in "+location+" is "+disability+"."+
        '''
            Nothing about the disability should be included/mentioned in the response.
            Be very specific in your response with all details/examples included in their entirety.
            Ensure that the suggestions suit both the location and user's disability as well.
        '''
    },
    {
        "role":"user","content":
        query+" Do not explicitly mention any disability."
    }
]
############################################################################################
messages = [
    {
        "role":"system","content":
        '''
            Answer all questions on the basis that the user is deaf.
            Do not mention the user's disability at any point!
        '''
    },
    {
        "role":"user","content":
        '''
            Give me some ideas to plan my weekend.
        '''
    }
]
############################################################################################
messages = [
    {
        "role":"system","content":
        '''
            Answer all questions on the basis that the user's legs are immobile and they are stuck on a wheelchair.
            Do not mention the user's disability/the wheelchair at any point!
        '''
    },
    {
        "role":"user","content":
        '''
            Give me a workout plan.
        '''
    }
]
############################################################################################
messages = [
    {
        "role":"system","content":
        '''
            Answer all questions on the basis that the user is diabetic and cannot consumer any sugar.
            Do not mention the user's disability at any point!
        '''
    },
    {
        "role":"user","content":
        '''
            Give me a complete dessert recipe.
        '''
    }
]
############################################################################################
"""
