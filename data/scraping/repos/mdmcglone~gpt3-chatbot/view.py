from flask import Flask, render_template
from flask_restful import Api, Resource
from flask_cors import CORS
import os
import openai

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

# Enable CORS -> Cross Origin Resource Sharing -> Allow requests from other domains, importantly for frontend javascript
CORS(app)


#decorator function that specifies the url suffix
@app.route('/')
def home():
    return render_template('index.html')
#renders basic html landing page

#Restful API endpoint
class chatbot(Resource):
    #query is now usable variable from url suffix
    def get(self, query):
        #prefix is the prompt for the GPT-3 model, which has the query appended to it
        prefix = 'you are a chatbot that...'
        query = prefix + query + '\n'
        try:
            #set api key for GPT-3 model 
            api_key = f'./secrets/API_KEY'
            api_key = open(api_key, 'r').read()
            openai.api_key = api_key

            #set parameters for gpt completion and pass query to model
            gpt = openai.Completion.create(
                prompt=query,
                engine="davinci",
                temperature=0.1, 
                max_tokens=256,
                top_p = 1,
                frequency_penalty=0,
                presence_penalty=0,
                stop = ['\n\nLawyer or Law Student:']
            )
            #return the response from the model
            pred = gpt.choices[0].text
            #format the response to remove the prefix and the prompt
            out = pred.split('Chatbot: ')[1]
            return {'response': out}
        except Exception as e:
            return {'error': str(e)}

#specifices name and url suffix of Restful API endpoint
api.add_resource(chatbot, '/<string:query>')

#this is the main function that runs the app
#THIS SHOULD NOT BE DEBUG MODE IN PRODUCTION
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)


