from flask import Flask, request, jsonify
import sys,os,logging, config
import pandas as pd
sys.path.insert(0,'../scripts/')
from cohere_news_score import cohere_classify
from data_fetch import get_news_data
logging.basicConfig(filename='../log/log.log', filemode='a',encoding='utf-8', level=logging.DEBUG)

api_key =config.cohere_api['api_key']
news = get_news_data(path='data/Example_data.xlsx',repo='C:/Users/User/Desktop/Prompt-Engineering',version='Example_data_v1')

app = Flask(__name__)

@app.route('/')
def index():
    
    return jsonify({
                "status": "success",
                "message": "Index Page"
             })
@app.route('/bnewscore', methods=['GET', 'POST'])
def news_score_route():
    if request.method == 'POST':
        body = request.get_json()['Body']
        prompt = list()
        prompt.append(body)
        output = cohere_classify(api_key,news,prompt)

        return jsonify({"status": "success", "news_item_score": output[2]})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 33507))
    app.run(host='0.0.0.0', debug=True, port=port)