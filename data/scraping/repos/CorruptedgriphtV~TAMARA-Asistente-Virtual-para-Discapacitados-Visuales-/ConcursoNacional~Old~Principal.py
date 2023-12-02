import cv2
import requests
import pyttsx3
import openai
import os


import pygame

def play_sound(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
def generate_text(prompt):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=100
    )
    #print("Respuesta de GPT-3:", response)  # Agrega esta línea para ver la respuesta en bruto
    generated_text = response.choices[0].text.strip()
    #print("Texto generado:", generated_text)  # Agrega esta línea para ver el texto generado
    return generated_text

api_key = "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"
openai.api_key = api_key

def main():
    # Configurar las credenciales de la API de GPT-3


    # Configurar la URL y las credenciales de la API de Computer Vision
    endpoint = "https://tamaraserver.cognitiveservices.azure.com/"
    subscription_key = "975da0dae182426ba219f03767e152ae"
    api_url = f"{endpoint}vision/v3.2/analyze"

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    while True:
        # Capturar la imagen desde la cámara
        ret, frame = cap.read()

        # Reproducir sonido intro antes de escuchar la solicitud
        play_sound("Intro.mp3")

        # Guardar la imagen en un archivo temporal
        img_temp_path = "temp_img.jpg"
        cv2.imwrite(img_temp_path, frame)

        # Leer la imagen como bytes
        with open(img_temp_path, "rb") as image_file:
            image_data = image_file.read()

        # Realizar la solicitud a la API de Computer Vision
        headers = {
            "Content-Type": "application/octet-stream",
            "Ocp-Apim-Subscription-Key": subscription_key
        }
        params = {
            "visualFeatures": "Description",
            "language": "es"  # Solicitar descripción en español
        }
        response = requests.post(api_url, headers=headers, params=params, data=image_data)

        # Procesar la respuesta de la API de Computer Vision
        if response.status_code == 200:
            result = response.json()
            if "description" in result and "captions" in result["description"]:
                description = result["description"]["captions"][0]["text"]

                # Agregar la descripción al archivo de registro en la carpeta Sandbox
                with open("temp_description.txt", "a") as f:
                    f.write(description + "\n")
            else:
                print("No se encontró una descripción en la respuesta de Azure.")
        else:
            print("Error en la solicitud a Azure:", response.text)

        # Eliminar el archivo temporal de la imagen
        if os.path.exists(img_temp_path):
            os.remove(img_temp_path)

        # Esperar la entrada del usuario
        user_input = input("Ingrese su solicitud: ")

        # Reproducir sonido outro después de recibir la solicitud
        play_sound("Outro.mp3")

        # Si el usuario pregunta qué está viendo o qué se ve
        if any(phrase in user_input.lower() for phrase in ["qué estoy viendo", "qué se ve", "que estoy viendo", "que se ve"]):
            # Hablar la descripción generada por Azure
            engine = pyttsx3.init()
            engine.setProperty('voice', 'spanish')  # Establecer idioma a español
            engine.say("Imagen reconocida: " + description)
            engine.runAndWait()

        # Generar respuesta adicional con GPT-3 utilizando la descripción almacenada en el archivo de registro
        with open("temp_description.txt", "r") as f:
            saved_description = f.read()
        prompt = "Eres TAMARA, asistes a personas con discapacidad visual y tu objetivo es proporcionar una descripción detallada de la imagen y ayudar en lo que el usuario solicite, no inventes muchas cosas, solo retroalimenta a lo que ves .\n\nUsuario: {}\n\nDescripción de Azure: {}\n\nRespuesta:"
        prompt_with_input = prompt.format(user_input, saved_description)
        additional_assistance = generate_text(prompt_with_input)
        # Generar respuesta adicional con GPT-3
        prompt = "Eres TAMARA, asistes a personas con discapacidad visual y tu objetivo es proporcionar una descripción detallada de la imagen y ayudar en lo que el usuario solicite, no inventes muchas cosas, solo retroalimenta a lo que ves .\n\nUsuario: {}\n\nDescripción de Azure: {}\n\nRespuesta:"
        prompt_with_input = prompt.format(user_input, description)
        additional_assistance = generate_text(prompt_with_input)

        # Imprimir y pronunciar la respuesta adicional generada por GPT-3
        print("Respuesta adicional generada por GPT-3:", additional_assistance)
        engine = pyttsx3.init()
        engine.setProperty('voice', 'spanish')  # Establecer idioma a español
        engine.say(additional_assistance)
        engine.runAndWait()

if __name__ == "__main__":
    main()
