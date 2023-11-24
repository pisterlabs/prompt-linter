# from flask import Flask, request, jsonify
# import os
# from apikey import apikey
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

# os.environ["OPENAI_API_KEY"] = apikey  # Fix the environment variable name

# app = Flask(__name__)

# @app.route("/create-user", methods=["POST"])
# def create_user():
#     data = request.get_json()

#     # Check if 'question' and 'answer' are present in the JSON data
#     if 'question' in data and 'answer' in data:
#         question = data['question']
#         answer = data['answer']

#         template = PromptTemplate(
#             input_variables=['question', 'answer'],  # Correct the field name to 'input_variables'
#             template="Evaluate answer and give feedback as a json format(only give json format evaluate answer. do not give any other explanations) \n give me your feedback using the following section in json `result(answer is correct or wrong)` `mistake`,`clarify (clarify the corrected answer relevant to the question)` ,`correction (if provided answer wrong ,correct that answer)`,`overall_feedback`, `marks(out of 10 , marks should be a number)`. don't use `:` symbols inside the object values  if the answer is not relevant for this question please mention that in this section \n Question: {question} \n Answer :{answer}"
#         )

#         llm = OpenAI(temperature=0.9)
#         chain = LLMChain(llm=llm, prompt=template, verbose=True)

#         response = chain.run(question=question, answer=answer)
#         print(response)
#         return response, 201
#     else:
#         return jsonify({"error": "Invalid input data. 'question' and 'answer' fields are required."}), 400

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'