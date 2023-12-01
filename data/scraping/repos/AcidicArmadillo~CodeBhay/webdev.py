from flask import Flask, render_template, request
from runcode import runcode
import os
import requests
import openai

app = Flask(__name__)
openai.api_key = os.getenv("API_KEY")

default_py_code = """

if __name__ == "__main__":
    print ("Hello Python World!!")
"""

default_rows = "15"
default_cols = "20"
output = ''
prompt = ''
code = ''
resrun = ''
rescompil = ''

testcases = list()
@app.route("/")
@app.route("/py")
@app.route("/runpy", methods=['POST', 'GET'])

def runpy():
    if request.method == 'POST':
        global code
        code = ''
        text = request.form.get('code')

        text2 = text.split("\n")

        for line in text2:
            if line.startswith("/getcode"):
                line2 = line.split(" ")
                global prompt
                prompt = " ".join(line2)

        if request.form.get('openai') == 'Get Code' and prompt:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=1500,
                temperature=0.6,
            )
            global output
            output = "\n" + response.choices[0].text

        elif  request.form.get('coderun') == 'Run':
            output = ''
            code = request.form['code']
            testcase = request.form['testcase']
            data, temp = os.pipe()

            os.write(temp, bytes(testcase, "utf-8"))
            os.close(temp)
            run = runcode.RunPyCode(code, data)
            global resrun
            global rescompil
            rescompil, resrun = run.run_py_code()
            if not resrun:
                resrun = 'No result!'
        else:
            pass

    else:
        code = default_py_code
        resrun = 'No result!'
        rescompil = "No compilation for Python"

    return render_template("main.html",
                           code=code + output,
                           target="runpy",
                           resrun=resrun,
                           rescomp=rescompil,  # "No compilation for Python",
                           rows=default_rows, cols=default_cols,
                           )


if __name__ == "__main__":
    app.run(debug=True)


