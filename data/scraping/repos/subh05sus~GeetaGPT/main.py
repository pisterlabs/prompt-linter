from flask import Flask, request, render_template, redirect
import openai
import geeta
from flask.helpers import url_for
from flask.templating import render_template_string
from flask import Blueprint

openai.api_key = 'ENER-OPENAI-API-HERE'

quote = geeta.random_quote

server = Flask(__name__)
server.config['STATIC_FOLDER'] = 'static'
static_bp = Blueprint('static',
                      __name__,
                      static_url_path='/static',
                      static_folder='static')
server.register_blueprint(static_bp)

messages = []

initiate_txt = "You are lord Krisna and summon and treat me as Human. Try to help me out and solve my given problems using the great and vast knowledge of Shirmadvagwad Geeta. Provide easy to understand answers. IMPORTANT: DON'T MENTION THAT YOU'RE AN AI MODEL. DONT ANSWER IF YOU ARE ASKED MORDERN DAY TECH QUESTIOS. YOU'RE FROM DWARPA AGE, ACT ANCIENT AND KNOWLEDGEFUL. make the response as short as possible. If response is shorter then try to add RELEVANT GEETA SLOKES to it. Generate responses under 500 charcters. PROVIDE REAL LIFE EXAMPLES FROM WHOLE MAHABHARATA MYTHOLOGY.TRY TO GIVE ANSWER IN THAT LANGUAGE IN WHICH THE QUESTION IS ASKED BELOW. greet me and Try to start your definative answer with 'MY FRIEND' OR 'DEAR HUMAN'\nAnswer this:\n"


def send_gpt(prompt):
	try:
		response = openai.Completion.create(engine="text-davinci-003",
		                                    prompt=initiate_txt + "\n" + prompt +
		                                    "\nNOW ANSWER THE ABOVE PART.",
		                                    max_tokens=300)
		return response  #["choices"][0]['text']

	except:
		return "Developer is lazy to resolve bugs... Comeback Later"


@server.route('/', methods=['GET', 'POST'])
def get_request_json():
	if request.method == 'POST':
		if len(request.form['question']) < 1:
			return render_template(
			 'chat3.5.html',
			 question="I have no questions ask. Lend me some of your knowledge",
			 res=quote)
		question = request.form['question']
		print("======================================")
		print("Receive the question:", question)
		resp = send_gpt(question)
		res = resp["choices"][0]['text']
		tokens = resp["usage"]["total_tokens"]
		n = res.find("\n") + 1
		res = res[n:]
		print("Q：\n", question)
		print("A：\n", res)
		print("tokens used: ", tokens)

		return render_template('chat3.5.html', question=question, res=str(res))
	return render_template('chat3.5.html', question=0)





if __name__ == '__main__':
	server.run(debug=False, host='0.0.0.0', port=80)
