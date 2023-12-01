import openai
import requests
import time

TOKEN = "5819512077:AAFRWnhLhGeYEJQqjcLKCFhr3z5jVgKUaEs"

def get_updates(offset=None):
    # definimos url
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    params = {'offset': offset, "timeout":100} 
    response = requests.get(url, params=params)
    return response.json()["result"]

def send_messages(chat_id, text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    params = {"chat_id": chat_id, "text":text}
    response = requests.post(url, params=params)
    return response
    
def get_openai_response(prompt):
    model_engine = "davinci:ft-arkusnexus-2023-07-15-01-10-02"
    response = openai.Completion.create(
        engine = model_engine,
        prompt = prompt,
        max_tokens = 250,
        temperature = 0,
        n = 1,
        stop = "END"
    )
    return response.choices[0].text.strip()

def main():
    print('Starting bot..')  # Imprime en la consola el mensaje "Starting bot..".
    offset = 0  # Inicializa una variable llamada offset con el valor 0.
    while True:  # Inicia un bucle infinito.
        # Llama a nuestra función actualizadora.
        updates = get_updates(offset)  # Llama a la función get_updates() para obtener actualizaciones de mensajes.
        if updates:  # Verifica si hay actualizaciones de mensajes disponibles.
            for update in updates:  # Itera sobre cada actualización de mensaje recibida.
                offset = update['update_id'] + 1  # Actualiza el valor de offset para evitar procesar la misma actualización nuevamente.
                chat_id = update["message"]["chat"]["id"]  # Obtiene el identificador de chat de la actualización de mensaje actual.
                user_message = update["message"]["text"]  # Obtiene el texto del mensaje enviado por el usuario.
                print(f"Received message: {user_message}")  # Imprime en la consola el mensaje recibido del usuario.
                GPT = get_openai_response(user_message)  # Llama a la función get_openai_response() para obtener una respuesta de OpenAI.
                send_messages(chat_id, GPT)  # Llama a la función send_message() para enviar la respuesta al usuario.
        else:  
            time.sleep(1)  # Pausa la ejecución del programa durante 1 segundo.

if __name__ == '__main__':
    main()  # Llama a la función main() para iniciar la ejecución del programa.