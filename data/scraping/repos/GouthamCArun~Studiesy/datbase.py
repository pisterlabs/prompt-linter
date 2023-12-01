import firebase_admin
from firebase_admin import credentials,firestore
import json
import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')

cred = credentials.Certificate("serviceAccount.json")
firebase_admin.initialize_app(cred) 
db=firestore.client()



def importing(subject):
    print(subject)
    list=db.collection("Notes").document(subject).get()
    if list.exists:  
        list=db.collection("Notes").document(subject).get()
        data=list.to_dict()
        print("done")
        return (data.get('summary', '').strip())
    else:
        return ("there was no period today")







# to generate summary
def to_summ(value,subject):
    reuslt=""
    while (value=="True"):
        list=db.collection("Notes").document(subject).get()
        data=list.to_dict()
        result1=""
        if 'summary' in data:
            result1=data["summary"]
            print(result1)
            
        else:
            def prompt():
                llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.1)

                prompt = PromptTemplate(
                    input_variables=["summ"],
                    template="write a detailed summary of {summ} and explain it in a very understandable way.",
                )
                chain = LLMChain(llm=llm, prompt=prompt)
                return chain

            def question1(inputs):
                chain = prompt()
                return chain.run(inputs)
            data1=question1(data['transcribe'])
            db.collection("Notes").document(subject).update({"summary":data1})
            result1=data1
            print(result1)
        value="False"
    return result1

