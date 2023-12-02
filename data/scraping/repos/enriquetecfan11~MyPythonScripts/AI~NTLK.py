import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('stopwords')

def resumir_texto(texto, cantidad_oraciones=3):
    # Tokenización de palabras y oraciones
    palabras = word_tokenize(texto)
    oraciones = sent_tokenize(texto)

    # Eliminación de stopwords
    palabras = [palabra.lower() for palabra in palabras if palabra.isalnum() and palabra.lower() not in stopwords.words('english')]

    # Cálculo de frecuencia de palabras
    frecuencia = FreqDist(palabras)

    # Obtención de las oraciones más relevantes
    oraciones_importantes = sorted(oraciones, key=lambda oracion: sum(frecuencia[palabra] for palabra in word_tokenize(oracion)))

    # Tomar las primeras 'cantidad_oraciones' oraciones
    resumen = ' '.join(oraciones_importantes[:cantidad_oraciones])

    return resumen

# Ejemplo de uso
texto_idea = "Una startup que permite a los usuarios aprender más sobre sus registros médicos, permitiéndoles hojearlos y encontrar información. Esto podría ser utilizado por consumidores para obtener más información sobre su salud, por médicos para obtener más información sobre sus pacientes y por compañías de seguros para obtener más información sobre sus clientes."

resumen_idea = resumir_texto(texto_idea)
print(resumen_idea)

import openai

# Configura tu clave de API de GPT-3
openai.api_key = "tu_clave_de_api"

def extender_con_gpt3(texto, temperatura=0.7, max_tokens=100):
    # Parámetros de la llamada a la API de GPT-3
    prompt = f"Extiende la siguiente idea:\n\n'{texto}'\n\nExtensión:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=temperatura,
        max_tokens=max_tokens,
    )

    # Extraer el texto extendido de la respuesta de GPT-3
    texto_extendido = response.choices[0].text.strip()
    return texto_extendido

# Ejemplo de uso
texto_idea = "Una startup que permite a los usuarios aprender más sobre sus registros médicos, permitiéndoles hojearlos y encontrar información. Esto podría ser utilizado por consumidores para obtener más información sobre su salud, por médicos para obtener más información sobre sus pacientes y por compañías de seguros para obtener más información sobre sus clientes."

texto_extendido = extender_con_gpt3(texto_idea)
print(texto_extendido)
