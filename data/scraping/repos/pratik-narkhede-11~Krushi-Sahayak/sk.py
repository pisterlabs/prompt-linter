from flask import Flask,render_template,url_for,redirect,request
import openai

app=Flask(__name__)
@app.route('/')
def welcome():
  return render_template('page1.html')

@app.route('/answer/<string:ans>/<string:q>')
def func(ans):
  print(ans)
  return render_template('page2.html',answ=ans)


@app.route('/submit',methods=['POST','GET'])
def submit():
  if request.method=='POST':
    # # question=request.form['textbox']
    # # print(question)
    x=request.form['speechtotext']
    print(x)

    openai.api_key = "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"
    prompt = x
    model = "text-davinci-002"
    response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.7,
    )
    answer = response.choices[0].text.strip()
    return redirect(url_for('func',ans=answer))