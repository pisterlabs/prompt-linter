from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from NLP.Anthropic import Anthropic
from NLP.OpenAI import OpenAI

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/document', methods=['POST'])
async def generate_doc():
    data = request.get_json()
    modelContent = data['prompt']
    prompt = await Anthropic.mk_prompt(modelContent)    
    response = await Anthropic.run_request(prompt)

    return response

@app.route('/generate', methods=['POST'])
async def generate_metric():
    data = request.get_json()
    modelContent = data['prompt']

    prompt1 = await Anthropic.promptDesignMetrics(modelContent)    
    response1 = await Anthropic.run_request(prompt1)
    #print(response1)
    metric_name, statement = response1.split(":")
    
    prompt2 = await Anthropic.promptGenMetrics(modelContent, statement, metric_name)
    response2 = await Anthropic.run_request(prompt2)
    text_list = response2.split('```')
    #print(text_list)
    yaml_text = text_list[1]
    explanation = text_list[2]
    response3 = yaml_text.replace("yaml\n", "")            
    #print(response3)
    
    return jsonify({'result': response3, 'explanation': explanation})

@app.route('/generateSQL', methods=['POST'])
async def generate_sql():
    data = request.get_json()
    modelContent = data['prompt']

    prompt = await Anthropic.promptGenMetricSQL(modelContent)    
    response = await Anthropic.run_request(prompt)    

    text_list = response.split("```")
    sql_text = text_list[1]
    responseF = sql_text.replace("sql\n", "")  
    
    return jsonify({'result': responseF})

if __name__ == '__main__':
    app.run()