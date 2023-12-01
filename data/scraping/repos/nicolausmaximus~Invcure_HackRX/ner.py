import spacy
from spacy import displacy
import openai
import environ
env = environ.Env()
environ.Env.read_env()

openai.api_key=env('OPENAI_API_KEY')

NER = spacy.load("en_core_web_sm")

def NamedER(text):
    
    text1= NER(text)
    print("The entities are:")
    # print(text)
    for word in text1.ents:
        print(word.text,word.label_)

def openaiNERPatientName(text):
    # print(text)
    # print("hello")
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt="Extract the Patient Name:"+text,
    temperature=0,
    max_tokens=200,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    # print("Patient Name is: "+response['choices'][0]['text'])
    return response['choices'][0]['text']

def openaiNERAddress(text):
    # print(text)
    # print("hello")
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt="Extract the Address:"+text,
    temperature=0,
    max_tokens=200,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    print("Address is: "+response['choices'][0]['text'])
    return response['choices'][0]['text']

def openaiNERDate(text):
    # print(text)
    # print("hello")
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt="Extract the Date:"+text,
    temperature=0,
    max_tokens=200,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    print("Date is: "+response['choices'][0]['text'])
    return response['choices'][0]['text']

def openaiNERGender(text):
    # print(text)
    # print("hello")
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt="Extract the Gender:"+text,
    temperature=0,
    max_tokens=200,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    print("Gender is: "+response['choices'][0]['text'])
    return response['choices'][0]['text']

def openaiNERPhoneNumber(text):
    # print(text)
    # print("hello")
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt="Extract the Phone Number:"+text,
    temperature=0,
    max_tokens=200,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    print("Phone Number is: "+response['choices'][0]['text'])
    return response['choices'][0]['text']

def openaiNEREmail(text):
    # print(text)
    # print("hello")
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt="Extract the Email:"+text,
    temperature=0,
    max_tokens=200,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    print("Email is: "+response['choices'][0]['text'])
    return response['choices'][0]['text']

def openaiNERAmount(text):
    # print(text)
    # print("hello")
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt="Extract the total amount:"+text,
    temperature=0,
    max_tokens=200,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    print("Total Amount is: "+response['choices'][0]['text'])
    return response['choices'][0]['text']

    
def openaiNERItems(text):
    # print(text)
    # print("hello")
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt="Extract all the items in invoice:"+text,
    temperature=0,
    max_tokens=200,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    print("The items are: "+response['choices'][0]['text'])
    return response['choices'][0]['text']

def extractAll(text):
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt="Extract Patient Name, Gender, Date, Amount, Email, Phone Number and Address:"+text,
    temperature=0,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    print("The recognized text is are: "+response['choices'][0]['text'])
    return response['choices'][0]['text']
