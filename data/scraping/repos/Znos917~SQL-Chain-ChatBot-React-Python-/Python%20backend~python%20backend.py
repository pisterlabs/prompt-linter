from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from flask import Flask,request,jsonify
from flask_cors import CORS
import environ
env = environ.Env()
environ.Env.read_env()
app = Flask(__name__)
CORS(app)
API_KEY = env('OPENAI_API_KEY')

# Setup database

@app.route('/first', methods=['POST'])
def first():
    db = SQLDatabase.from_uri("sqlite:///Chinook_Sqlite.sqlite")
    # setup llm
    llm = OpenAI(temperature=0, openai_api_key=API_KEY)
    # Create db chain
    QUERY = """
    Given an input question, first create a syntactically correct sqlite query to run, then look at the results of the query and return the answer.
    Use the following format:

    Question: Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here

    {question}
    """
    # Setup the database chain
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
    data=request.get_json()
    prompt=data['prompt']
    question = QUERY.format(question=prompt)
    response=db_chain.run(question)
    return jsonify({"response":response})
    print(response)
if __name__ == '__main__':
    app.run()
    
