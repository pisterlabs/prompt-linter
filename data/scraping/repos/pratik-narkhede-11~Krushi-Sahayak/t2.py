import googletrans
from googletrans import Translator
from googletrans import LANGUAGES
from flask import Flask,render_template,url_for,redirect,request
import openai

app=Flask(__name__)
@app.route('/')
def welcome():
  return render_template('index.html')

@app.route('/answer/<string:ans>/<string:q>')
def func(ans,q):
  return render_template('index2.html',answ=ans, qe=q)


@app.route('/submit',methods=['POST','GET'])
def submit():
  if request.method=='POST':
    # # question=request.form['textbox']
    # # print(question)
    x=request.form['speechtotext']
    print(x)
    
    translator = Translator()
    # unknown_sentence = x
    results = translator.detect(x)
    orignal = x
    translation = translator.translate(x, dest='en')
    x=translation.text
    fl=0
    if(results.lang == 'mr'):
      fl=1
      print(results.lang)
      print(fl)
    elif(results.lang == 'hi'):
      fl=2
      print(results.lang)
    elif(results.lang == 'en'):
      fl=0
      print(results.lang)
    

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
    
    if(fl==1):
        translator = Translator()
        user_output = translator.translate(answer, dest='mr')
        output_var= user_output.text
        
    elif(fl==2):
        translator = Translator()
        user_output = translator.translate(answer, dest='hi')
        output_var= user_output.text
        
    elif(fl==0):
        output_var= answer

    
    return redirect(url_for('func',ans=output_var, q=orignal))


if __name__=='__main__':
    app.run(debug=True)