from time import time,sleep
import openai
from flask import Flask, request, redirect

convo = []
ticktock = 1
max_length = 20
with open('openaiapikey.txt', 'r') as infile:
    open_ai_api_key = infile.read()
openai.api_key = open_ai_api_key


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()



app = Flask(__name__)


def gpt3_completion(prompt, engine='text-curie-001', temp=1.0, top_p=1.0, tokens=200, freq_pen=1.0, pres_pen=1.0, stop=['User:','EVE:']):
    max_retry = 5
    retry = 0
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            filename = '%s_gpt3.txt' % time()
            with open('gpt3_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(0.25)


def generate_html(convo):
    html = '<html><head></head><body>'
    html += '<form action="/" method="post">'
    html += '<input type="text" name="message" autofocus="autofocus">'
    html += '<input type="submit">'
    html += '</form>'
    html += '<ul>'
    for message in convo:
        html += '<li>{}</li>'.format(message)
    html += '</ul>'
    html += '</body></html>'
    return html


def get_chatbot_response(convo):
    global ticktock
    ticktock *= -1
    if ticktock > 0:
        prompt_file = 'eve_ask.txt'
    else:
        prompt_file = 'eve_idea.txt'
    prompt = open_file(prompt_file)
    convo_text = ''
    for i in convo:
        convo_text += i + '\n'
    convo_text = convo_text.strip()
    prompt = prompt.replace('<<CONVO>>', convo_text)
    return gpt3_completion(prompt)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.form['message']
        convo.append('User: %s' % message)
        #print(convo)
        response = get_chatbot_response(convo)
        convo.append('EVE: %s' % response)
        return redirect('/')
    if len(convo) >= max_length:
        a = convo.pop(0)
    return generate_html(convo)


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.run(debug=True)