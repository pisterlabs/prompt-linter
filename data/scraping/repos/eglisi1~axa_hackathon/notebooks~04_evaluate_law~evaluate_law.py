import json
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

openai.api_key = os.environ['OPENAI_API_KEY']

input = {
    "beteiligter": "Beteiligter 1",
    "fahrzeug": "Ford Fiesta",
    "aktionen": [
        {
            "id": 1,
            "beschreibung": "Hat rechts abgebogen",
            "artikel": {
                "VRV 28 1": "Der Fahrzeugführer hat alle Richtungsänderungen anzukündigen, auch das Abbiegen nach rechts. Selbst der Radfahrer, der zum Überholen eines andern ausschwenkt, hat dies anzuzeigen.",
                "VRV 17 5": "Kündigt der Führer eines Busses im Linienverkehr innerorts bei einer gekennzeichneten Haltestelle mit den Richtungsblinkern an, dass er wegfahren will, so müssen die von hinten herannahenden Fahrzeugführer nötigenfalls die Geschwindigkeit mässigen oder halten,"
            }
        },
        {
            "id": 2,
            "beschreibung": "Hat nicht geblinkt",
            "artikel": {
                "VRV 28 1": "Der Fahrzeugführer hat alle Richtungsänderungen anzukündigen, auch das Abbiegen nach rechts. Selbst der Radfahrer, der zum Überholen eines andern ausschwenkt, hat dies anzuzeigen.",
                "VRV 17 5": "Kündigt der Führer eines Busses im Linienverkehr innerorts bei einer gekennzeichneten Haltestelle mit den Richtungsblinkern an, dass er wegfahren will, so müssen die von hinten herannahenden Fahrzeugführer nötigenfalls die Geschwindigkeit mässigen oder halten,"
            }
        }
    ],
}

action_list = ''

for aktion in input['aktionen']:
    action_list += aktion['beschreibung'] + ', '

violation_schema = ResponseSchema(name='violation',
                                  description='Wurde gegen den Gesetzesartikel verstossen? \
                             Antworte True wenn ja, oder antworte False wenn Nein oder es nicht klar ist.')

reason_schema = ResponseSchema(name='reason',
                               description='Begründe warum gegen den Gesetzesartikel \
                             verstossen wurde oder warum nicht.')

response_schemas = [violation_schema,
                    reason_schema]

violation_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = violation_output_parser.get_format_instructions()

violation_prompt_template = PromptTemplate(
    input_variables=['aktionsliste', 'artikel'],
    template="""\
        Evaluiere anhand der folgenden kommaseparierten Liste von Aktionen ob die Person gegen den folgenden Gesetzesartikel verstösst des folgenden  ob die folgende aktion gegen ihn verstösst und \
        extrahiere die dazu folgenden Informationen: \
        
        Violation: Wurde gegen den Gesetzesartikel verstossen? Antworte True \
        wenn ja, oder antworte False wenn Nein oder es nicht klar ist.gegen den folgenden Gesetzesartikel '{artikel}' verstösst.",
        
        Reason: Begründe warum gegen den Gesetzesartikel verstossen wurde oder warum nicht.
    
        Aktionsliste: {aktionsliste}
        
        Artikel: {artikel}
        
        Formattiere die Antwort in ein valides JSON.
    """
)

for aktion in input['aktionen']:
    prompt = violation_prompt_template.format(aktionsliste=action_list, artikel=aktion['artikel_text'])

    llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=violation_prompt_template, verbose=True)

    output = chain.run({'aktionsliste': action_list, 'artikel': aktion['artikel_text']}, )

    output_dict = json.loads(output)

    print(output_dict)
