from pymongo.mongo_client import MongoClient
import openai
from flask import Flask
from flask import jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allows flask app to not have network restrictions

# vv Example data set from Database vv
'''
data = {
  "_id": "RingCentral",
  "Benefit": "Description",
  "Dental Insurance": "Offered by employer",
  "Free Lunch": "5 days a week",
  "Health Insurance": "Offered by employer",
  "Life Insurance": "Offered by employer",
  "Vision Insurance": "Offered by employer",
  "Health Savings Account (HSA)": "Offered by employer",
  "Maternity Leave": "Offered by employer",
  "Sick Time": "Unlimited",
  "Roth 401k": "Offered by employer",
  "Employee Stock Purchase Program (ESPP)": "Allows contributions up to 15% of base salary. 10% discount on purchase price of stock",
  "Donation Match": "100% match. Up to $1,000 matched",
  "Flexible Spending Account (FSA)": "Offered by employer",
  "Disability Insurance": "Offered by employer",
  "401k": "50% match on the first 6% of base salary",
  "Remote Work": "Depends on your manager, team, and needs.",
  "Paternity Leave": "Offered by employer",
  "PTO (Vacation / Personal Days)": "Unlimited",
  "Employee Discount": "Offered by employer"
}
'''

# Post method of our custom python api (gets data previously stored in DataBase)
@app.route('/db-up/company', methods=['POST'])
def input_mongo():

    # Connection details
    uri = "mongodb+srv://VenkatSagi:mongodb.2004@cluster0.ijkgw8w.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri)
    db = client.WorkerBenefitsDB
    global coll
    coll = db.WorkerBenefitsCo

    # API Key
    openai.api_key = "API_KEY"

    # Get company data
    global companyIn
    companyIn = input("What Company do you work for: ")

# Post method of our custom python api (gets input and runs through chatGPT to get response)
@app.route('/gpt-up/user', methods=['POST'])
def input_chat():
    user = input("How may I be of service: ") #initial user input/ querey question.
    global completion
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": f"Provides any specifications the user has related to their employee benefits! "
                        f"The benefits are related to the company \"{companyIn}\". {companyData}"},
            {"role": "user", "content": user}
        ]
    )  # open ai compltetions api
    return jsonify(message="Chat Output Successful"), 200  # returns message client side

# Get method for our custom Python Api (Gets ChatGPT response from the completion)
@app.route('/gpt-down/user', methods=['GET'])
def get_chatResponse():  # method calls and returns transcribed text client side
    return completion.choices[0].message.content  # prints out Open Ai response

# Get method for our custom Python Api (Gets json response from MongoDB and checks for existense)
@app.route('//db-down/company', methods=['GET'])
def get_json():  # method calls and returns transcribed text client side

    # Store the json data from company name (checks from MongoDB)
    global companyData
    companyData = coll.find_one({"_id": companyIn})

    # Check if company exists
    if companyData is None:
        return("Company does not exist")
        exit(1)

    # Default return
    return companyData


if __name__ == '__main__':  # port this server is running on
    app.run(port=5000)
