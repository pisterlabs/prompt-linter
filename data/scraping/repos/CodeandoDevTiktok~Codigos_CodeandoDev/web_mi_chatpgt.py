from flask import Flask, render_template, request, session
import openai
from datetime import datetime

# Configura la clave de la API
openai.api_key = "pon-tu-api-key"

# Configura la aplicación Flask
app = Flask(__name__)
app.secret_key = 'clave_codeando_3'

# Función de chat
def enviar_conversacion(mensajes):
    # Crear una nueva lista de mensajes con solo las propiedades "role" y "content"
    mensajes_api = [{"role": msg["role"], "content": msg["content"]} for msg in mensajes]

    respuesta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mensajes_api
    )
    return respuesta.choices[0].message.content

# Ruta principal
@app.route('/', methods=['GET', 'POST'])
def index():
    mensajes = []  # Variable local para almacenar los mensajes
    
    if 'mensajes' in session:
        mensajes = session['mensajes']
    
    if request.method == 'POST':
        mensaje = request.form['mensaje']
        tiempo = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        mensajes.append({"role": "user", "content": mensaje, "time": tiempo, "name": "Usuario"})
        respuesta = enviar_conversacion(mensajes)
        
        tiempo = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        mensajes.append({"role": "assistant", "content": respuesta, "time": tiempo, "name": "Asistente"})
        
    session['mensajes'] = mensajes  # Actualizar los mensajes en la sesión
    
    return render_template('index.html', mensajes=mensajes)

    
if __name__ == '__main__':
    app.run(debug=True)