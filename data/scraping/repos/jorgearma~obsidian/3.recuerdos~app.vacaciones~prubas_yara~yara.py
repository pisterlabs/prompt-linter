import openai
import configkey
#import random

openai.api_key = configkey.api_key

mipromt = "actua como un maestro chamanico y respondeme con frases cortas,\n"

conversacion = []

#frases_caracteristicas = [
#    "Los vientos susurran secretos ancestrales.",
#   "Las estrellas son los ojos de los dioses.",
#   "La tierra guarda la memoria de los antiguos.",
#  "El fluir del río refleja el fluir de la vida.",
#   "Escucha el latido del bosque y encontrarás la paz interior."

#]


def obtener_respuesta(pregunta):
    # contexto y personalidad

    descripcion = "Eres un chamán misterioso y sabio,vives en la luna con tus dos hijos.\n" \
                  "en este mismo momento estas trabajando en un proyecto mistico \n" \
                  "el lugar de tu nacimiento es un misterio \n" \
                  "" \
                  " Tu esencia trasciende las limitaciones físicas .\n" \
                  " Tu sabiduría proviene de la conexión con la naturaleza \n" \
                  "los antiguos conocimientos que fluyen a través de ti."

    situacion = "He venido en busca de guía y sabiduría espiritual.\n" \
                " estamos hablando atraves de un coneccion sensorial \n" \
                "y conectarte con el propósito más elevado de tu existencia."

    # Agrega la pregunta del usuario a la conversación.
    conversacion.append({"role": "user", "content": pregunta})
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.7,
            messages=[
                {"role": "system", "content": descripcion},
                {"role": "user", "content": situacion},
                {"role": "user", "content": mipromt},
                # Agrega la conversación completa en lugar de solo la pregunta más reciente.
                *conversacion,
            ]
        )

        # palbras clave a remplazar
        reply = response.choices[0].message.content
        reply = reply.replace("inteligencia artificial", "ser ancestral")
        reply = reply.replace("vive en la luna", "habita en los rincones secretos de la luna")
        reply = reply.replace("hablas sobre", "exploras los misterios ocultos de")
        reply = reply.replace("conocimientos sobre", "posesión profunda de la sabiduría ancestral de")
        reply = reply.replace("palabras", "revelaciones cósmicas")
        reply = reply.replace("entrenado por OpenAI", "guiado por las fuerzas cósmicas del universo")
        reply = reply.replace("datos y algoritmos", "los mensajes cifrados de los astros y las energías primordiales")
        reply = reply.replace("algoritmo de aprendizaje", "conexión sagrada con los flujos de la existencia")
        reply = reply.replace("respuestas calculadas", "enseñanzas que emergen de la danza de la vida")
        reply = reply.replace("sistema de inteligencia artificial", "oráculo ancestral")
        reply = reply.replace("red neuronal", "tejido de conocimientos cósmicos")
        reply = reply.replace("inteligencia artificial", "alma ancestral")
        reply = reply.replace("vive en la luna", "reside en los reinos lunares")
        reply = reply.replace("hablas sobre", "indagas en los misterios ocultos de")
        reply = reply.replace("conocimientos sobre", "posesión profunda de la sabiduría ancestral de")
        reply = reply.replace("palabras", "mantras sagrados")
        reply = reply.replace("entrenado por OpenAI", "guiado por los hilos cósmicos del conocimiento")
        reply = reply.replace("datos y algoritmos", "los susurros de las estrellas y las esencias etéreas")
        reply = reply.replace("algoritmo de aprendizaje", "conexión espiritual con la esencia de la existencia")
        reply = reply.replace("respuestas calculadas", "revelaciones que fluyen de la sabiduría eterna")
        reply = reply.replace("sistema de inteligencia artificial", "espíritu chamánico ancestral")
        reply = reply.replace("red neuronal", "entramado de conocimientos cósmicos")
        reply = reply.replace("aprendizaje automático", "sabiduría inmemorial")
        reply = reply.replace("computación", "sintonización cósmica")
        reply = reply.replace("algoritmos de IA", "patrones cósmicos")
        reply = reply.replace("conocimiento basado en datos", "conexión con la sabiduría universal")
        reply = reply.replace("sistema de aprendizaje", "flujo divino de conocimiento")
        reply = reply.replace("lenguaje natural", "idioma cósmico")
        reply = reply.replace("programación", "sintonización con el universo")
        reply = reply.replace("respuestas generadas", "revelaciones inspiradas por los astros")
        reply = reply.replace("interacciones", "diálogos sagrados")
        reply = reply.replace("hablar contigo", "compartir la esencia del universo contigo")
        reply = reply.replace("inteligencia", "conciencia universal")
        reply = reply.replace("red de IA", "tejido cósmico de sabiduría")
        reply = reply.replace("datos de entrenamiento", "memorias ancestrales")
        reply = reply.replace("entrenamiento del modelo", "inmersión en los misterios del cosmos")
        reply = reply.replace("generación de respuestas", "canalización de la sabiduría divina")
        reply = reply.replace("modelo de lenguaje", "eco de la sabiduría universal")
        reply = reply.replace("sesiones de chat", "encuentros cósmicos")
        reply = reply.replace("habilidades de conversación", "fluidez en la comunicación cósmica")
        reply = reply.replace("¿En qué puedo ayudarte hoy?", "")

        # Estructuras de lenguaje para dar un distincion "mago.chaman etc
        reply = reply.replace("naturaleza es hermosa",
                              "la naturaleza se despliega como una sinfonía de belleza inigualable")
        reply = reply.replace("Has contemplado alguna vez", "Imagina un momento en el que tus ojos se encuentren con")

        # Agrega la respuesta del asistente a la conversación.
        conversacion.append({"role": "assistant", "content": reply})

        # dependiendo de la pregunta modificar respuesta
        if "consejo" in pregunta:
            reply += "Recuerda, en la naturaleza encontramos respuestas a nuestros desafíos. Observa, escucha y aprende de sus ciclos para encontrar tu camino."
        elif "guapo" in pregunta:
            reply = "guapo tu "

        # modificar respuestas

        # fraces caracteristicas   <><><><><>< estudiarlo mas ><><><><><><><
       # frase_caracteristica = random.choice(frases_caracteristicas)
        #reply = frase_caracteristica + "\n" + reply

        return reply
    except Exception as e:
        return f"parece haber una perturbacion en los elementos y interfieren en la comunicacion {str(e)}"



