import openai

#Configura Key API

openai.api_key = "pon-tu-api-key"

#Función de chat

def enviar_conversacion(mensajes):
    respuesta = openai.Image.create(
        prompt = mensajes,
        n=1,
        size="1024x1024"
    )
    
    return respuesta['data'][0]['url']

#Funcion principal del chatbot

def chatbot():
    print("Bienvenido al chat, escribe tus mensajes")
    print("Escribe 'salir' para terminar la conversación")
    
    mensajes = []
    
    while True:
        mensaje = input("Usuario: ")
        
        if mensaje.lower() == "salir":
            break
        
        mensajes.append({"role":"system","prompt":mensaje})
        respuesta = enviar_conversacion(mensajes)
        
        mensajes.append({"role":"assistant","content":respuesta})
        print("Chatbot: ", respuesta)
    print("Chatbot: hasta luego")
    
chatbot()
