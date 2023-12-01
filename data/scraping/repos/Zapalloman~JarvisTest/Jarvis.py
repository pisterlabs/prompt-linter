import speech_recognition as sr
import openai

# Configuración de la API de OpenAI
openai.api_key = ''

# Función para reconocer la voz
def reconocer_voz():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Di algo...")
        audio = r.listen(source)
        
    try:
        texto = r.recognize_google(audio, language='es')
        print("Has dicho: " + texto)
        return texto
    except sr.UnknownValueError:
        print("No se pudo reconocer la voz.")
    except sr.RequestError as e:
        print("Error en la solicitud al servicio de reconocimiento de voz: {0}".format(e))

# Función para generar respuestas utilizando GPT-3.5
def generar_respuesta(texto):
    respuesta = openai.Completion.create(
        engine="davinci-codex",
        prompt=texto,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )
    return respuesta.choices[0].text.strip()

# Bucle principal del asistente virtual
def iniciar_asistente():
    while True:
        entrada = reconocer_voz()
        if entrada:
            if entrada.lower() == "salir":
                print("Hasta luego.")
                break
            respuesta_gpt = generar_respuesta(entrada)
            print("Asistente: " + respuesta_gpt)

# Iniciar el asistente virtual
iniciar_asistente()
