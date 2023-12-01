import json, io, base64
import gradio as gr
import openai
import requests
import gradio as gr 
from config import API_KEY_OPENAI , URL_MODEL, API_HF

openai.api_key = API_KEY_OPENAI

def traducir(text):
    return openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        max_tokens = 100,
        messages = [{
    "role": "user",
    "content": f"resumir en no mas de 3 palabras y traducir al español y unir las palabras con guines bajos:{text}"
        }],
        temperature = 1).choices[0].message["content"]


def genera_titulo_de_imagen(inputs, model, api, parameters=None):
    """
    Genera un título para una imagen utilizando un modelo de generación de texto.

    Parámetros:
    - inputs: Los datos de entrada necesarios para generar el título de la imagen.
    - model: La URL o endpoint del modelo de generación de texto.
    - api: La clave de autorización necesaria para acceder al modelo.
    - parameters: Parámetros adicionales opcionales para personalizar la generación del título.

    Retorna:
    - Un objeto JSON que contiene el título generado para la imagen.

    """
    headers = {
        "Authorization": f"Bearer {api}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                model,
                                headers=headers,
                                data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))

def _imagen_de_base64_str(pil_image):
    """
    Convierte una imagen PIL en una cadena base64.

    Parámetros:
    - pil_image: La imagen PIL a convertir.

    Retorna:
    - Una cadena base64 que representa la imagen convertida.

    """
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))

def _titulo(imagen):
    """
    Genera un título para una imagen utilizando una imagen y funciones auxiliares.

    Parámetros:
    - imagen: La imagen para la cual se generará el título.

    Retorna:
    - El título generado para la imagen, después de traducirlo.

    """
    base64_image = _imagen_de_base64_str(imagen)
    result = genera_titulo_de_imagen(base64_image,
                                     model=URL_MODEL,
                                     api=API_HF)
    text = result[0]['generated_text']
    return traducir(text)
        
if "__main__" == __name__:
    gr.close_all()
    demo = gr.Interface(fn=_titulo,
                    inputs=[gr.Image(label="Cargar Imagen", type="pil")],
                    outputs=[gr.Textbox(label="Titulo para la imagen")],
                    title="Generedor de nombre para imagenes",
                    description="Generedor de nombre para imagenes",
                    allow_flagging="never",
                    examples=["christmas_dog.jpeg", "bird_flight.jpeg", "cow.jpeg"])

    demo.launch()