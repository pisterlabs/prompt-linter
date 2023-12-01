from flask import Flask, request, jsonify, render_template
import os
import openai
import json

# create the flask app
app = Flask(__name__)
openai.api_key = os.environ['OPEN_AI_KEY']
# what html should be loaded as the home page when the app loads?
@app.route('/')
def home():
    rand_text = os.environ['TEXT']
    return render_template('app_frontend.html', prediction_text=rand_text)

# define the logic for reading the inputs from the WEB PAGE, 
# running the model, and displaying the prediction
@app.route('/predict', methods=['GET','POST'])
def predict():

    # get the description submitted on the web page
    begin = "My second grader asked me what this passage means:\n\"\"\"\n"
    end = "\n\"\"\"\nI rephrased it for him, in plain language a second grader can understand:\n\"\"\"\n"
    video_text = request.form.get('description')
    entire_text = begin + video_text + end
    response = openai.Completion.create(
        engine="davinci",
        prompt= entire_text,
        temperature=0.3,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        stop=["\n"]
    )
    print(entire_text)
    return render_template('app_frontend.html', prediction_text=response.choices[0].text)
    #return 'Description entered: {}'.format(a_description)

#@app.route('/prediction', methods=['GET', 'POST'])
#def prediction():
#    if request.method == 'POST':
#        prediction_data = request.json
#        print(prediction_data)
#    return jsonify({'result': prediction_data})

# boilerplate flask app code

HOST = '0.0.0.0'
PORT = 5000
if __name__ == "__main__":
    context = ('server.crt', 'server.key')
    app.run(host=HOST, port=PORT, threaded=True, debug=True, ssl_context=context)
    #app.run(debug=True)
