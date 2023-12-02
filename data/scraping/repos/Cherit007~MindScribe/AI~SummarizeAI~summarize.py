from flask import Flask ,request , jsonify
import cohere
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(api_key)

app=Flask(__name__)
@app.route("/summarize",methods=["POST"])
def summary():
    req=request.get_json()
    texts=req["text_to_summarize"]
    response = co.summarize(
                text=texts,
                )
    return jsonify({"summarized_text":response.summary})


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=int("9070"),debug=True)
