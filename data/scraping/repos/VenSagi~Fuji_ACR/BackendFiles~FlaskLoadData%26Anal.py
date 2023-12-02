from pymongo.mongo_client import MongoClient
from bson.objectid import ObjectId
import openai
from flask import Flask, request, abort
from flask import jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # allows flask app to not have network restrictions


# Get method of our custom python api (gets data previously stored in DataBase and puts in chatgpt)
@app.route('/db-up/gpt', methods=['GET'])
def input_chat():
    # Mongodb  server variables
    uri = "mongodb+srv://VenkatSagi:mongodb.2004@cluster0.ijkgw8w.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri)
    db = client.ToyotaRaces
    collection = db.DragStripRaces

    # Mongodb database variables
    raceID = request.args.get('raceID')
    raceInfo = collection.find_one({'_id': ObjectId(raceID)})
    vehicleModel = raceInfo['Vehicle']

    # ChatGPT Api Key
    openai.api_key = "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"

    # Getting Chat Completion
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": f"Provides recomendations for the user, based on their Drag Strip Time slip info provided."
                        f"Gives specific recomendations to improve performance."
                        f"Recomends what the user need the most improvement on (analize time slip stats)."
                        f"Specifies how (be specific on how) they can improve it."
                        f"Give recomendations of potential car parts, modifications, drills, etc."
                        f"Mentions the user's (before) stats and tells them potential (after) stats with specific improvements."
                        f"Tells them how much time they can save with specific modifications, drills, etc."
                        f"Gives result in bullet point format."
                        f"Time Slip & Data: {raceInfo}"},
            {"role": "user", "content": f"Give me specific recomendations to improve performance for my {vehicleModel}"}
        ]
    )  # open ai compltetions api
    return jsonify(message=completion.choices[0].message.content), 200


@app.route('/db-up/raceid', methods=['POST'])
def input_raceID():
    if not request.json or 'raceID' not in request.json:
        abort(400)
    raceID = request.json['raceID']
    return jsonify({'raceID': raceID}), 201



if __name__ == '__main__':  # port this server is running on
    app.run(port=5000)
