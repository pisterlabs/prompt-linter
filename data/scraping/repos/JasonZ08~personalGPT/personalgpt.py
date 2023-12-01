import os
import sys
import openai
import constants 
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from flask import Flask, request
import pickle
import sqlite3

app = Flask(__name__)

@app.route("/chat", methods=["POST"])

def chat(): 
    def append_to_file(statement):

        #add data to text file
        try:
            file = open("./data/data.txt", "a")
            file.write(statement + "\n")
            file.close()
            return "Input successfully added to the file."
        except:
            return "There was something wrong appending your message to the text file."

    def execute_sql_command(statement):

        #connect to database and execute the command
        conn = sqlite3.connect("./database/calendar.db")
        cur = conn.cursor()

        try:
            cur.execute(statement)
            conn.commit()
            cur.close()
            conn.close()
            return "SQL statement executed successfully."
        except:
            return "There is something wrong with your SQL insert statement."

    def query_txt(query):

        #run the query taking in the personal text file
        loader = DirectoryLoader("./data", glob="*.txt")
        index = VectorstoreIndexCreator().from_loaders([loader])

        return index.query(query, llm=ChatOpenAI())

    def query_db(query):

        #connect to the database
        db = SQLDatabase.from_uri("sqlite:///database/calendar.db")
        llm = ChatOpenAI()
        db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

        #run the query taking in the personal database
        return db_chain.run(query)

    #Get the message from the POST request
    query = request.get_json()
    print(query)

    #OpenAI API Key Configuration
    os.environ["OPENAI_API_KEY"] = constants.APIKEY
    openai.api_key = os.getenv("OPENAI_API_KEY")

    colon_index = query.find(":")
    if colon_index != -1 and query[:colon_index] == "SQLITEINPUT":

        #call the execute sql function
        answer = execute_sql_command(query[colon_index+1:].strip())
    elif colon_index != -1 and query[:colon_index] == "TEXTINPUT":

        #open the file and append data to it
        answer = append_to_file(query[colon_index+1:].strip())
    else:
        #get the model from the pickle file
        model_pkl_file = "database_text_question_classifier.pkl"
        with open(model_pkl_file, 'rb') as file:  
            model = pickle.load(file)

        #get the tokenizer from the pickle file
        token_pkl_file = "tokenizer.pkl"
        with open(token_pkl_file, 'rb') as file:  
            tokenizer = pickle.load(file)

        max_length = 30

        #have the tokenizer convert the input into numeric data
        inp = [query]
        sequences = tokenizer.texts_to_sequences(inp)
        inp = pad_sequences(sequences, maxlen=max_length)

        #predict whether the question wants data from database vs text file
        res = model.predict(inp)
        print(res)
        if res[0][1] > res[0][0]:
            #model predicts its a database question
            answer = query_db(query)
        else:
            #model predicts its a text question
            answer = query_txt(query)

    
    print(answer)

    return {"response": answer}

if __name__ == "__main__":
    app.run(debug=True)