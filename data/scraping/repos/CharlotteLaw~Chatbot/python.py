import openai
from flask import Flask, render_template, request
import mysql.connector
from html import unescape

#from api_secrets import API_KEY

app = Flask(__name__)

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="xxx", #Use your own MySQL password.
    database="messages" 
)

mycursor = db.cursor()

#Create your own database and table in MySQL Workbench.
#mycursor.execute("CREATE DATABASE messages")
#mycursor.execute("CREATE TABLE convo (ID int primary key, question VARCHAR(3000), response VARCHAR(3000))")
#set ID to auto increment

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def getvalue():
    
    mycursor.execute("SELECT question FROM convo ORDER BY ID DESC LIMIT 1")
    qu = str(mycursor.fetchall())
    qu = qu.replace("[('", '')
    qu = qu.replace("',)]", '')

    mycursor.execute("SELECT response FROM convo ORDER BY ID DESC LIMIT 1")
    re = str(mycursor.fetchall())
    re = re.replace("[('", '')
    re = re.replace('[("', '')
    re = re.replace("',)]", '')
    re = re.replace('",)]', '')
    num_appear = re.count('\\n')
    re = re.replace("\\n", '', num_appear)

    message = request.form['message']
    askprompt = "You are an informative, happy, and helpful personal assistant/ AI chatbot. Chat with the user or answer their questions. The conversation starts now. " + message
    openai.api_key = "xxx" #generate your own API key in OpenAi.
    response = openai.Completion.create(engine="text-davinci-001", prompt=askprompt, max_tokens=200)
    resp = response.choices[0].text
    print(response)
    mycursor.execute("INSERT INTO convo (question, response) VALUES (%s, %s)", (message, resp))
    db.commit()
    return render_template('index.html', m=message, r=resp, n=qu, s=re)


if __name__ == '__main__':
    app.run(debug=True)

