
import os
import openai
import json
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


# Now you can use os.getenv to access your variables
openai.api_key  = os.environ['OPENAI_API_KEY']
 
# Text-Input
situation1 = "G1, G2 und B1 fuhren in genannter Reihenfolge mit ca. 50 km/h vom Beschleunigungsstreifen Bolligen herkommend auf dem Normalstreifen der A6-Süd R. B2 fuhr kurz vor dem Unfall vom Beschleunigungsstreifen Wankdorf herkommend mit ca. 50 km/h auf die A6-Süd R. G1 und G2 mussten, weil sie an ein Stauende heranfuhren, bis zum Stillstnand abbremsen. B1 bemerkte aufgrund fehlender Aufmerksamkeit das Stauende nicht und konnte in der Folge nicht rechtzeitig Abbremsen. Er kollidierte gegen das heck von G2 und schob G2 gegen das Heck von G1. B2 nahm den Unfall zu spät war und kollidierte gegen das Heck von B1. Im Nachgang wurde festgestellt, dass B2 über keinen gültigen Führerausweis verfügt und unter Drogeneinfluss stand."
situation2 = "G kam über die Obere Zollgasse her gefahren, weiter in Richtung Ostermundigen. Auf Höhe der Verzweigung Waldheimstrasse kam gleichzeitig B über die Waldheimstrasse her gefahren und wollte nach rechts in die Obere Zollgasse einbiegen. B missachtete dabei den Vortritt und kollidierte so mit seiner linken Fahrzeugfront mit der rechten Fahrzeugseite von G. "
situation3 = "G1 kam von Hinterkappelen und wollte über die Neue Murtenstrasse in Richtung Bern fahren. B1 fuhr bei der Ausfahrt Bern-Bethlehem ab und wollte über die Neue Murtenstrasse in Richtung Bethlehem/Brünnen fahren. Bei der Kreuzung Eymattstrasse/Neue Murtenstrasse missachtete B1 das rote Lichtsignal und nahm damit G1 den Vortritt. In der Folge kollidierte G1 frontal mit der rechnten Seite des Fahrzeuges von B1."
situation4 = "B fuhr in Frauenkappelen von Mühleberg herkommend in Richtung Bern auf der Murtenstrasse. B fuhr mit ca. 45 km/h (eigene Angaben) B versuchte während der Fahrt Mücken abzuwehren und wendete dadurch die Aufmerksamkeit gegenüber der Strasse ab. G parkierte seinen Anhänger korrekt. Anschliessend kollidierte B mit der rechten Ecke der Front ihres Pw's gegen den korrekt auf der Fahrbahn parkierten Anhänger von G. Duch den Unfall wurden die Airbags des Pw B ausgelöst."

# Step 4: Define the Prompt Template
prompt = PromptTemplate(
    input_variables=["concept"],
    template= """You're a traffic specialist app, that gets information for a traffic accident. This is what happend {concept}, return only a python dictionary for every involved party with the strict following structure, seperate every involved party with a |: 
    element 1 = "beteiligter": "Sample Name",element 2 "fahrzeug": "Vehicle", element 3"aktionen": as a list that contains max. 4 objects "id": 1, "beschreibung": "Sample Description,v max. 10 words per aktion""",
)
 
# Step 5: Print the Prompt Template
print(prompt.format(concept=situation2))
 
# Step 6: Instantiate the LLMChain
llm = OpenAI(temperature=0.0, model_name="gpt-3.5-turbo")
chain = LLMChain(llm=llm, prompt=prompt, verbose = True)
 
# Step 7: Run the LLMChain
output = chain.run(situation2)
print(output)

if '|' in output:
    print("Chain-Output: " + output)
else:
    # Wenn nicht, wird eine Exception geworfen
    raise ValueError("Keine Trennung der Objekte durch | vorhanden")

splitted_list = output.split('|')
print("splittet_list") 
print(splitted_list)

# Wandeln Sie jeden String in der Liste in ein Dictionary um
dict_list = [json.loads(s) for s in splitted_list]

# dict_list ist nun eine Liste von Dictionaries
print("dict_list") 
print(dict_list)

dict_list[0]






