import openai
openai.api_key="INGRESE-SU-KEY"

def generar_dialogo(cond1, cond2, personaje1, personaje2):
    

    # Funcion para obtener la respuesta de la API de OpenAI
    def get_completion(prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.8, # this is the degree of randomness of the model's output
        )
        return response.choices[0].message["content"]

    prompt= f""" Instrucciones:
    1. Crear un generador de dialogos entre dos personajes de la base de datos. 
    2. El usuario debe elegir dos personajes que se encuentre en la base de datos.
    3. El usuario Puedes establecer parámetros para el tono de la conversación 
    (serio, humorístico, tenso, etc.) y la longitud de los diálogos generados. 

    El usuario pondra estas preferiencias {cond1}, {cond2}
    El usuario escoge los personajes {personaje1}, {personaje2}

    Cond1: tipo de humor que tendra el dialogo (humoristico, serio, tenso, etc.)
    Cond2: longitud del dialogo (cantidad de caracteres)

    Esto podría dar lugar a conversaciones segun cond1 y cond2 entre personajes,
    dados por personaje1 y personaje2.  

    Respuesta:
    Dialogo generado: 
    Obs: No quiero que la respuesta diga cuentos caracteres se han generado o algo relacionado a esto (cond2)
    Obs: Lo personjes deben ser los que el usuario escogio (personaje1 y personaje2)
    """

    # Obtener la respuesta de la API de OpenAI
    response = get_completion(prompt)
    return response


# print('Ingrese el tipo de humor que tendra el dialogo:')
# cond1 = input("Ejemplo: Humoristico, serio, tenso, etc. ")
# cond2 = input("Ingrese la longitud del dialogo: ")

# z = generar_dialogo(cond1, cond2)
# print(z)

