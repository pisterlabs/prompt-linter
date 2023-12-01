from flask import Flask, request
import openai
import os

openai.api_key = "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" 

app = Flask(__name__)


dialogs = ""
messages = []

@app.route('/')
def index():
    global dialogs, messages

    prompt = request.args.get("prompt", "")

    if prompt != "" :
        messages.append({"role": "user", "content": prompt})
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        res = completion.choices[0].message['content'].replace("\n", "<br/>").replace(" "," &nbsp;" )
        messages.append({"role": 'assistant', "content": res}  )

        dialogs += f'<div style="margin:20px 0px">🍳{prompt}</div>' 
        dialogs += f'<div style="background-color:#ddd;margin:20px 2px">😊{res}</div>'         

    html= f"""
        <div style="background-color:gray">{dialogs}</div>
        <form action=/> 
            <textarea style="width:100%"  rows=4 name=prompt></textarea>
            <input type=submit value=Chat>
        </form>
    """    
    return html


    
if __name__ == '__main__':
	app.run(debug=True)
