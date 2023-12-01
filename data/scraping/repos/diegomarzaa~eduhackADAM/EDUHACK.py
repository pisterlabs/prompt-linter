from openai import OpenAI
import time
import tkinter as tk
import requests
from PIL import Image
from io import BytesIO
from PIL import ImageTk
from configparser import ConfigParser

MODELO_CHAT = "gpt-3.5-turbo"
MODELO_IMAGENES = "dall-e-3"

# API KEY

api = ""  # Put your API key here if you don't want to be asked every time, keep the "".

if api == "":
    api = input("Enter your API key: ")
    api = api.strip()

client = OpenAI(api_key=api)


class Chat:
    def __init__(self, system_prompt):
        # Recoger respuesta del usuario
        self.user_input = ""

        # Sitio donde guardarlo rapidamente
        self.last_response = ""
        
        # HISTORY
        self.messages = [
                {"role": "system", "content": system_prompt}
            ]
        
        # Para evitar que se envíen mensajes mientras se está generando la respuesta
        self.chatting = False

        self.chat_square = None
        self.image_window = None


    def get_input(self, user_input):
        """
        Cogemos la box de tk y usamos el get para obtener el texto que hay en ella.
        """
        self.user_input = user_input.get()
        self.actualizar_chat_UI(f'User: {self.user_input}\n')
        print(self.user_input)


    def delete_input(self, user_input):
        """
        Cogemos la box de tk y usamos el delete para borrar el texto que hay en ella.
        """
        user_input.delete(0, tk.END)
        print("Input deleted.")


    def actualizar_chat_UI(self, palabra):
        # Change the configuration of a text item according to options.
        self.chat_square.insert(tk.END, palabra)
        # print("Chat history updated.")


    def generate_and_stream_response(self):
        """
        Generamos la respuesta a partir del input del usuario.
        """
        print("Generating response...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
            max_tokens=1500,
            stream=True
        )
    
        self.actualizar_chat_UI("Assistant: ")

        for chunk in response:
            # Comprobamos si la respuesta ha acabado
            if chunk.choices[0].delta.content == None:
                self.chatting = False
                break
            else:
                palabra = chunk.choices[0].delta.content
                self.last_response += palabra
                # print("Respuesta:", self.last_response)
                self.actualizar_chat_UI(palabra)
                self.chat_square.update()
    
        
        self.actualizar_chat_UI("\n")
        
        # print("Response generated:\n")
        # print(self.last_response)
    

    def add_user_response_object(self):
        """
        Añadimos la respuesta del usuario a la lista de mensajes.

        Limpiamos el user_input para que no se acumule.
        """
        self.messages.append({"role": "user", "content": self.user_input})


    def add_assistant_response_object(self):
        """
        Añadimos la respuesta del asistente a la lista de mensajes.
        """
        self.messages.append({"role": "assistant", "content": self.last_response})
    
    def clear_data(self):
        """
        Limpiamos el user_input para que no se acumule.
        """
        self.user_input = ""
        self.last_response = ""



    def create_image_prompt(self):
        """
        Crea un prompt para generar una imagen a partir de la última respuesta del asistente.
        """
        # Mostramos los primeros 10 caracteres
        print("\n######## Creando prompt de el siguiente concepto: " + self.last_response[:10] + "...")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a image prompt creator"},
                {"role": "user", "content": "Create a detailed prompt for an image based on the next concept, be concise and only show the prompt: " + self.last_response}
            ],
            max_tokens=500,
        )

        prompt = response.choices[0].message.content
        # Remove newlines
        prompt = prompt.replace("\n", "")
        print("\n######### Prompt creado:", prompt)
        return prompt

        

    def create_image_from_prompt(self, prompt):
        print("Generando imagen...")
        response_image = client.images.generate(
            model=MODELO_IMAGENES,   # dall-e-2, dall-e-3
            prompt=prompt,   # See revised_prompt # I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS
            size="1024x1024", # 256x256, 1024x1024, 1024x1792 or 1792x1024
            # quality="standard", # standard, hd
            n=1,  # In dalle2, you can add up to 10
            # style= "vivid" # vivid, natural     # Only for dalle3
        )
        print("Imagen generada.")
        
        image_url = response_image.data[0].url
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        while True:
            try:
                # Save in a subfolder in the same directory
                name = prompt + ".png"
                img.save(name)
                break
            except:
                prompt = prompt[:-1]

        print("Image saved.")
        return name
    

    def add_image_to_window(self, imagepath, window):
        # Si hay una imagen, la borramos
        for child in window.winfo_children():
            child.destroy()

        img = Image.open(imagepath)
        img = img.resize((500, 500))
        img = ImageTk.PhotoImage(img)
        label = tk.Label(window, image=img)
        label.image = img
        label.pack()
        print("Image added to window.")



    def create_questions_from_prompt(self, prompt="Abejas"):
        """
        Crea un cuestionario a partir de un prompt.
        """

        print("Generando cuestionario...")
        self.messages.append({"role": "user", "content": prompt})

        response_questions = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
            max_tokens=1000,
        )

        questions = response_questions.choices[0].message.content
        print("Cuestionario generado:", questions)
        return questions
    

    def put_questions_in_window(self, questions, window):

        # Si ya hay cuestionario, lo borramos
        for child in window.winfo_children():
            child.destroy()

        label = tk.Label(window, text=questions)
        label.config(font=("Arial", 13))
        # Change line when reaching the end of the window
        label.config(wraplength=300)
        label.pack()
        print("Questions added to window.")


    def prueba(self, user_input):
        """
        Prueba de concepto para ver si funciona el chat.
        """

        self.get_input(user_input)
        self.delete_input(user_input)
        self.add_user_response_object()
        self.generate_and_stream_response()
        # self.stream_response(response)
        self.add_assistant_response_object()


        # IMAGENES
        prompt = self.create_image_prompt()
        image_path = self.create_image_from_prompt(prompt)
        self.add_image_to_window(image_path, self.image_window)

        # CUESTIONARIO
        questions = cuestionario_chat.create_questions_from_prompt(self.last_response)
        cuestionario_chat.put_questions_in_window(questions, cuestionario_window)

        self.clear_data()


if __name__ == "__main__":
    main_chat = Chat("Eres un tutor educativo. Respondes a las preguntas de forma clara y concisa.")

    # VENTANA
    root = tk.Tk()
    root.title("Chat Application")
    root.geometry("800x500+520+100")
    root.resizable(width=tk.FALSE, height=tk.FALSE)

    # Conversación
    main_chat.chat_square = tk.Text(root)
    main_chat.chat_square.pack()
    main_chat.chat_square.config(width=80, height=18, font=("Arial", 15))

    # Input y botón
    user_input = tk.Entry(root)
    user_input.pack(side=tk.LEFT)
    user_input.focus_set()
    user_input.config(width=80, font=("Arial", 12))
    user_input.config(highlightbackground="black", highlightthickness=1)


    send_button = tk.Button(root, text="Enviar", command=lambda: main_chat.prueba(user_input))
    user_input.bind("<Return>", lambda event: main_chat.prueba(user_input))
    send_button.pack(side=tk.RIGHT)
    send_button.config(width=20, height=5, font=("Arial", 12), bg="white", fg="black")


    # SECONDARY WINDOW, IMAGES
    image_window = tk.Toplevel(root)
    image_window.title("Pestaña de imágenes")
    image_window.geometry("500x500+1315+100")
    image_window.resizable(width=tk.FALSE, height=tk.FALSE)

    # Image label
    image_label = tk.Label(image_window, text="")
    image_label.pack()
    main_chat.image_window = image_window


    # THIRD WINDOW, QUESTIONS
    cuestionario_chat = Chat("Eres un experto generando un ejercicio tipo test con respuestas multiples a partir de un texto. Limitate a ello, sin añadir nada más. Hazlo lo más simple y corto posible.")

    cuestionario_window = tk.Toplevel(root)
    cuestionario_window.title("Preguntas")
    cuestionario_window.geometry("500x500+20+100")
    cuestionario_window.resizable(width=tk.FALSE, height=tk.FALSE)

    root.mainloop()
