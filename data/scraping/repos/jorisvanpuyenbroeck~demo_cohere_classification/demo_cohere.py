import cohere
from flask import Flask, jsonify, request, render_template, redirect
from cohere.responses.classify import Example
import json

app = Flask(__name__)

json_file_path = 'config/cohere.json'

with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

api_key = data['api_key']

question=""
response=""

def askCohere(question):
  co = cohere.Client(api_key)
  examples = [
      Example("Hoe kan ik een kapotte lamp in de gang laten repareren?", "Melding probleem niet dringend"),
      Example("Er is een lekkage in mijn appartement. Help alsjeblieft snel!", "Melding dringend"),
      Example("Kun je me vertellen hoe ik de thermostaat in mijn appartement kan instellen?", "Vraag om technische informatie"),
      Example("Wanneer moet ik de maandelijkse servicekosten betalen?", "Vraag om financiële informatie"),
      Example("Is het mogelijk om mijn parkeerplaats te wijzigen?", "Melding probleem niet dringend"),
      Example("Er is een gaslek in de gemeenschappelijke ruimte. Dit is dringend!", "Melding dringend"),
      Example("Hoe kan ik mijn verwarmingssysteem in mijn appartement onderhouden?", "Vraag om technische informatie"),
      Example("Kun je me uitleggen hoe de servicekosten worden berekend?", "Vraag om financiële informatie"),
      Example("Mijn buren maken te veel lawaai. Kun je er alsjeblieft iets aan doen?", "Melding probleem niet dringend"),
      Example("Er is een waterlekkage in mijn badkamer. Kom zo snel mogelijk!", "Melding dringend"),
      Example("Hoe kan ik de airconditioning in mijn appartement efficiënt gebruiken?", "Vraag om technische informatie"),
      Example("Wat zijn de kosten voor het gebruik van de gemeenschappelijke wasruimte?", "Vraag om financiële informatie"),
      Example("Ik heb een kapotte deur in mijn appartement. Kan dit worden gerepareerd?", "Melding probleem niet dringend"),
      Example("Er is een brand in het gebouw! Bel onmiddellijk de brandweer!", "Melding dringend"),
      Example("Hoe kan ik mijn internetverbinding in mijn appartement verbeteren?", "Vraag om technische informatie"),
      Example("Kun je me uitleggen hoe de kosten voor gemeenschappelijke voorzieningen worden verdeeld?", "Vraag om financiële informatie"),
      Example("Mijn parkeerplaats is bezet door iemand anders. Wat moet ik doen?", "Melding probleem niet dringend"),
      Example("Er is een stroomstoring in het gebouw. We hebben snel hulp nodig!", "Melding dringend"),
      Example("Hoe kan ik mijn intercomsysteem instellen?", "Vraag om technische informatie"),
      Example("Wat zijn de regels voor huisdieren in het complex?", "Vraag om financiële informatie"),
      Example("Ik heb last van een lekkend plafond in mijn slaapkamer. Graag repareren!", "Melding probleem niet dringend"),
      Example("Er is een gaslek in de kelder. Dit is gevaarlijk, kom meteen!", "Melding dringend"),
      Example("Hoe kan ik mijn boiler onderhouden?", "Vraag om technische informatie"),
      Example("Wat zijn de kosten voor het gebruik van het zwembad?", "Vraag om financiële informatie"),
      Example("Mijn buurman speelt harde muziek 's nachts. Kan hier iets aan gedaan worden?", "Melding probleem niet dringend"),
      Example("Er is een waterlek in de gemeenschappelijke gang. Reparatie is dringend nodig!", "Melding dringend"),
      Example("Hoe kan ik mijn rookmelders testen?", "Vraag om technische informatie"),
      Example("Kun je me vertellen hoe de huurprijzen worden berekend?", "Vraag om financiële informatie"),
      Example("Mijn balkonhek is kapot. Kan dit worden gerepareerd?", "Melding probleem niet dringend"),
      Example("Er is een inbraak in mijn appartement geweest. Bel onmiddellijk de politie!", "Melding dringend"),
      Example("Hoe kan ik mijn verwarmingsthermostaat vervangen?", "Vraag om technische informatie"),
      Example("Wat zijn de kosten voor het gebruik van de fitnessruimte?", "Vraag om financiële informatie"),

  ]

  class Classification:
      def __init__(self, prediction, confidence, labels):
          self.prediction = prediction
          self.confidence = confidence
          self.labels = labels

  response = co.classify(  
      model='embed-multilingual-v2.0',  
      inputs=[question],  
      examples=examples)

  return response


@app.route('/', methods=['GET'])
def index():
    global question, response

    return render_template('index.html', question=question, response=response)

@app.route('/response', methods=['POST'])
def response():
  question = request.form['question']
  response = askCohere(question)

  # Redirect to the response page after processing the form
  return render_template('response.html', question=question, prediction=response[0].prediction, confidence=response[0].confidence)

@app.route('/reset', methods=['POST', 'GET'])
def reset():
    global question, response
    question = ""
    response = ""
    return redirect("/")

if __name__ == '__main__':
    app.run(debug=True)



