import openai

import os

from flask import Flask, render_template_string, request


openai.api_key = os.environ['OPENAI_API_KEY']


def generate_tutorial(components):

 response = openai.ChatCompletion.create(

  model="gpt-3.5-turbo",

  messages=[{

   "role": "system",

   "content": "You are a helpful Sports Coach who is the best coach in the world"

  }, {

   "role":

   "user",

   "content":

   f"You are a world class sports coach with knowledge of all games played on the planet. The user is a newbie and wants to know all the rules associated with the game. Please help him out by telling him all the rules associated with {components} game so that he can play the game accordingly."

  }])

 return response['choices'][0]['message']['content']


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])



def hello():

 output = ""

 if request.method == 'POST':

  components = request.form['components']

  output = generate_tutorial(components)


 return render_template_string('''

 <!DOCTYPE html>

 <html>

 <head>

  <title>SportsRule Website</title>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">

  <script>

  async function generateTutorial() {

   const components = document.querySelector('#components').value;

   const output = document.querySelector('#output');

   output.textContent = 'Writing all the rules for you...';

   const response = await fetch('/generate', {

    method: 'POST',

    body: new FormData(document.querySelector('#tutorial-form'))

   });

   const newOutput = await response.text();

   output.textContent = newOutput;

  }

  function copyToClipboard() {

   const output = document.querySelector('#output');

   const textarea = document.createElement('textarea');

   textarea.value = output.textContent;

   document.body.appendChild(textarea);

   textarea.select();

   document.execCommand('copy');

   document.body.removeChild(textarea);

   alert('Copied to clipboard');

  }

  </script>

 </head>

 <body>

  <div class="container">

   <h1 class="my-4">SportsCoach Game Rules Generator</h1>

   <form id="tutorial-form" onsubmit="event.preventDefault(); generateTutorial();" class="mb-3">

    <div class="mb-3">

     <label for="components" class="form-label">Name of game/sport:</label>

     <input type="text" class="form-control" id="components" name="components" placeholder="Enter the name of sport to generate rules for. For ex. Chess,etc " required>

    </div>

    <button type="submit" class="btn btn-primary">Share the rules with me</button>

   </form>

   <div class="card">

    <div class="card-header d-flex justify-content-between align-items-center">

     Output:

     <button class="btn btn-secondary btn-sm" onclick="copyToClipboard()">Copy</button>

    </div>

    <div class="card-body">

     <pre id="output" class="mb-0" style="white-space: pre-wrap;">{{ output }}</pre>

    </div>

   </div>

  </div>

 </body>

 </html>

 ''',

                output=output)


@app.route('/generate', methods=['POST'])


def generate():

 components = request.form['components']

 return generate_tutorial(components)


if __name__ == '__main__':

 app.run(host='0.0.0.0', port=8080)