import os
from apikey import apikey
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ['OPENAI_API_KEY'] = apikey
app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_request():
    data = request.get_json()
    cuisine = data.get('cuisine')
    ingredients = data.get('ingredients')
    

    llm = ChatOpenAI(temperature=0.6,model_name="gpt-3.5-turbo-16k")
    prompt_template_name = PromptTemplate(
        input_variables=['combined_input'],
        template = 'I want to cook a food using {combined_input} , what can I cook? Give me 1 meal with the list of neccesarry ingredients and cooking steps.'
    )
    chain = LLMChain(llm = llm , prompt = prompt_template_name)
    combined_input = f"{ingredients} from the {cuisine} cuisine"
    result = chain.run(combined_input)

    return jsonify({"result": result})


CORS(app, resources={r"/process": {"origins": "http://localhost:3000"}})
if __name__ == '__main__':
    app.run(debug=True,port=5001)

