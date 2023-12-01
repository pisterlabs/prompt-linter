import openai

openai.api_key = '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'

def obtener_contexto(ejercicio):
    # Esta función utiliza la API de OpenAI para obtener el contexto de un ejercicio
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Identifica el concepto principal o la acción a realizar en el siguiente ejercicio de programación en términos de conceptos fundamentales (tipos de datos primitivos, manejo de variables, operaciones con cadenas y listas, principios de diseño, estructuras de control, operaciones con colecciones, indexación y manejo de archivos)."},
                  {"role": "user", "content": ejercicio}]
    )
    return response.choices[0].message['content']

def generar_ejercicios(ejercicio,contexto):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "En base al siguiente contexto genera un ejercicio para personas que recien esten aprendiendo python, toma como ejemplo el siguiente ejercicio"},
                  {"role": "user", "content": "ejercicio: "+ejercicio+" y el contexto: "+contexto}]
    )
    return response.choices[0].message['content']

# Ejemplo de uso
ejercicio = input("Ingrese un ejercicio: ")
contexto = obtener_contexto(ejercicio)
ejercicio_similar = generar_ejercicios(ejercicio,contexto)
print(f"Contexto: {contexto}")
print(f"Ejercicio Similar: {ejercicio_similar}")
