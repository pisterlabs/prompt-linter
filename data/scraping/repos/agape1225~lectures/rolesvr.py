from flask import Flask, request
import openai
import os

api_key = os.environ["api_key"]
openai.api_key = api_key 

app = Flask(__name__)


dialogs = ""
messages = []

@app.route('/')
def index():
     html = """
        <a href=/talk?role=라푼젤><img src=/static/라푼젤.jpg width=150 ></a> <br/>
        <a href=/talk?role=백설공주><img src=/static/백설공주.jpg width=150 ></a> <br/>
     """ 
     return  html


@app.route('/talk')
def talk():
    global dialogs, messages

    role = request.args.get("role")    
    prompt = request.args.get("prompt", "")

    if prompt == "" :
        messages = []
        dialogs = ""
        messages.append({"role":"system","content":  f"당신은 친절한 {role} 입니다.가능한 모든 질문에 친절하게 답해주세요 "})
        messages.append({"role" :"user", "content": "당신은 누구 인가요?"})
        messages.append({"role":"assistant","content":  f"저는 {role} 입니다. 저에게 궁금한점을 물어보세요."})

    else :
        messages.append({"role": "user", "content": prompt})
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        res = completion.choices[0].message['content'].replace("\n", "<br/>").replace(" "," &nbsp;" )
        messages.append({"role": 'assistant', "content": res}  )

        dialogs += f'<div style="margin:20px 0px">🍳{prompt}</div>' 
        dialogs += f'<div style="background-color:#ddd;margin:20px 2px"><img src=/static/{role}.jpg width=30 >{res}</div>'         

    html= f"""
        <div style="background-color:gray">{dialogs}</div>
        <form action=/talk> 
            <input type=hidden name=role value={role}>
            <textarea style="width:100%"  rows=4 name=prompt></textarea>
            <input type=submit value=Chat>
        </form>
    """    
    return html


    
if __name__ == '__main__':
	app.run(debug=True)