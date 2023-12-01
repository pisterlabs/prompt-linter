from flask import Flask, request, render_template
import warnings
import cohere

warnings.filterwarnings("ignore")

app = Flask(__name__)
@app.route('/')
@app.route('/home')
def home():
    return render_template("main.html")

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        data = request.form
        print(data)
#///////////////////////////// LLM Code ///////////////////////////////
        co = cohere.Client('OjjnLULlAuvZBqBFNr3CUmreGvKwOP0UnAFMhgmH')
        name = data['name'] #"Akarsh"
        job_role= data['job-role'] #"software engineer"
        recipient_name = data['recipient-name'] #"Lakshmi"
        Recipient_Position = data['position'] #"HR"
        Company_Name = data['company'] #"HNR"
        job_description = data['description'] # write code and execute
        prompt_out = f'Consider you are a professional currently working in a MNC with sufficient experience. write a cover letter in that standard for me. My name is {name}. I am applying for the role of {job_role} in {Company_Name}. The recipent of the letter is {recipient_name}. The recipent designation is {Recipient_Position}. The job description for which i am applying is {job_description}.It should be catchy and simple. Display only the cover letter content.'
        response = co.generate(
        prompt= prompt_out,
        )
        for i in response:
            a = i
        # print(name,job_description,job_role,recipient_name,Recipient_Position,Company_Name)
        # a = data['name']
#///////////////////////////// LLM Code ///////////////////////////////

        return render_template('result.html',data=a)


if __name__ == '__main__':
    app.run()
