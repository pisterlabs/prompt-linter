from flask import Flask, request, jsonify
import sys,os,logging
sys.path.insert(0,'../scripts/')
from cohere_entity_extraction import cohere_extractor
from preprocessor import input_preprocessor
from data_fetch import get_job_data
import config 

logging.basicConfig(filename='../log/log.log', filemode='a',encoding='utf-8', level=logging.DEBUG)
api_key =config.cohere_api['api_key']
data = get_job_data(path='data/relations_dev.txt',repo='C:/Users/User/Desktop/Prompt-Engineering',version='relations_dev.txt_v1')
processed_data = input_preprocessor(data.head(n=8))

app = Flask(__name__)

@app.route('/')
def index():
    
    return jsonify({
                "status": "success",
                "message": "Index Page"
             })
@app.route('/jdentities', methods=['GET', 'POST'])
def jobs_route():
    if request.method == 'POST':
        prom = request.get_json()['prompt']
        prom = str(prom)+'\n\nout put:'
        extractor = cohere_extractor(api_key,processed_data,prom)
        return jsonify({"status": "success", "extracted_entities": extractor.replace('--end--','').replace('\n','  ')})
    if request.method == 'GET':
        return jsonify({
                "status": "success",
                "Training_data": processed_data,
             })
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 33507))
    app.run(host='0.0.0.0', debug=True, port=port)

