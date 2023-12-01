import openai
import time

openai.api_key = "INSERTA TU TOKEN AQUI"

def chatbot_response(prompt):
    completions = openai.Completion.create(
        engine='text-davinci-002',
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = completions.choices[0].text
    return message.strip()

def tipografia_escritura(texto, retraso=0.05):
    for caracter in texto:
        print(caracter, end='', flush=True)
        time.sleep(retraso)
    print('')

tipografia_escritura('Hola. Soy una Inteligencia Artificial.')
tipografia_escritura('¿De qué te gustaría hablar conmigo?')
contexto = ""

while True:
    pregunta = input("Tu: ")
    if pregunta.lower() in ['salir', 'terminar', 'exit']:
        break
    respuesta = chatbot_response(contexto + pregunta)
    contexto += respuesta
    tipografia_escritura(respuesta)