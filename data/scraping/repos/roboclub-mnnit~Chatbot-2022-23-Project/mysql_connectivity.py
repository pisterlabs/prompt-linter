import mysql.connector
import openai
import os
from dotenv import load_dotenv
load_dotenv()
def DataUpdate(query,answer):
    mydb=mysql.connector.connect(
        host="localhost",
        user="root",
        passwd=os.getenv('password'),
        database="chatbot"
    )
    mycursor=mydb.cursor()
    # openai.api_key=os.getenv('api')
    # completions=openai.Completion.create(engine='text-davinci-002',prompt=query,max_tokens=250)
    # message=completions.choices[0].text
    # message2=message.splitlines()
    # answer=message2[-1]
    ins='INSERT INTO fallback (Query,Answer) VALUES("{}","{}");'.format(query,answer)
    mycursor.execute(ins)
    mydb.commit()
    # print("printing message")
    # print(message)
    # print(message2)
    # print(answer)
    print(mycursor._rowcount,"record inserted")

# if __name__=="__main__":
#     DataUpdate("sarthak","sk")