from flask import Flask, request
from flask_cors import CORS, cross_origin
import cohere
import json
from api1 import doIt, findInts
import os
from twilio.rest import Client
app = Flask(__name__)
CORS(app)

print('test1')


@app.route("/hhh", methods=["POST", "GET"])
@cross_origin()
def getText():
    if request.method == "POST":
        txt = request.get_json('txt')
        city = doIt(txt['txt'])
        num = findInts(txt['txt'])
        print(city, num)
        # print(num)
        dict = {'city': city, 'num': num, 'txt': txt['txt']}
        with open("../shelter-finder/src/components/data/data.json", "w") as outfile:
            json.dump(dict, outfile)
        return {'city': city, 'num': num, 'txt': txt['txt']}
    if request.method == "GET":
        return 'blah'


account_sid = 'ACf5a3595d60c9a49607efe47bf3b3102d'
auth_token = '377ff4f8b56332cbed779b35ff0d2018'
client = Client(account_sid, auth_token)


@app.route("/bbb", methods=["POST", "GET"])
@cross_origin()
def getText2():
    if request.method == "POST":
        with open("../shelter-finder/src/components/data/data.json", "r") as outfile:
            data = json.load(outfile)
        message = client.messages.create(
            body=data['txt'],
            from_='+15673443322',
            to='+17788867253'
        )
        print(message.sid)
        print('success')
    return message.sid


if __name__ == "__main__":
    app.run(debug=True)
