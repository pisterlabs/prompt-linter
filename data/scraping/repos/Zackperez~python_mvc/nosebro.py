import nlpcloud
import os
import openai

token = "0c763b98f814c4649754c8c6e50425f99969aa72"
client = nlpcloud.Client("python-langdetect", token)
# Returns a json object.

def existe_lenguaje(idioma):
    for indice in range(len(idioma)):
        b = idioma[indice]
        if b in lista_lenguajes:
            print("idioma detectado: "+idioma)
            b = ""
        else:
            print("El idioma no existe")    

lang = client.langdetection("Hard to love")
espanol = "es"
lista_lenguajes = ["es", "de", "ru", "pt", "ko", "ja","en","sl"] #español,aleman,ruso,portugues,coreano,japones,inglés

a = lang.get('languages')
listapy = a[0]
listafinal = listapy.items()
idioma = list(listafinal)[0][0]
existe_lenguaje(idioma)
#for a in range(len(idioma)):
#    print(a)
print("Tamaño",len(idioma))

"""  
tamano = len(idioma)
existe_lenguaje(idioma)
print(lang)
print(tamano)
"""  

"""  
# -*- coding: utf-8 -*-
import openai
import json

openai.api_key =("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")
humano_respuestas = []
ia_respuestas = []

def existe_historial():
    try:
        with open('answer.json') as archivo:
            return True
    except FileNotFoundError as e:
        return False

def devolver_respuestas(humano_respuestas, ia_respuestas):
    respuestas = {}
    respuestas = {"Humano":humano_respuestas, "IA":ia_respuestas}
    return respuestas
    
def crear_answer_json():

    if existe_historial() == True:
        nuevos_datos = devolver_respuestas(humano_respuestas, ia_respuestas)
        with open("answer.json",encoding="utf-8") as archivo_json:
            datos = json.load(archivo_json)
        datos.append(nuevos_datos)

        with open("answer.json", 'w',encoding="utf-8") as archivo_json:
            json.dump(datos, archivo_json, indent=3, ensure_ascii=False)
            print("Se han añadido los siguientes datos al archivo " + archivo_json.name+"\n")
            print(datos)
    else:
        with open("answer.json", 'w',encoding="utf-8") as archivo_json:
            historial = []
            historial.append(devolver_respuestas(humano_respuestas, ia_respuestas))
            json.dump(historial, archivo_json, indent=3, ensure_ascii=False)
            print(archivo_json.name+" creado exitosamente")
            print("Se han añadido los siguientes datos al archivo " + archivo_json.name+"\n")
            print(historial)

conversation ="Fui creado por OpenAI. ¿Cómo te puedo ayudar hoy?"
print(conversation)

i = 1
while (i !=0):
    question = input("Human: ")
    if question == "Adios":
        respuestas = devolver_respuestas(humano_respuestas, ia_respuestas)
        crear_answer_json()
        print("AI: ¡Adiós!")
        break
    humano_respuestas.append(question)
    conversation += "\nHuman:" + question + "\nAI:"
    response = openai.Completion.create(
        model="davinci",
        prompt = conversation,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=["\n"," Human:", " AI:"]
    )

    answer = response.choices[0].text.strip()
    ia_respuestas.append(response.choices[0].text)

    conversation += answer
    print("AI: "+ answer)



    root = tk.Tk()
    app = Controller(root)
    root.mainloop()
import tkinter as tk

import tkinter as tk
import json
import nlpcloud
from tkinter import ANCHOR, ttk
import os
import openai

class Model:

    def __init__(self):
        self.texto_traducir = tk.StringVar()

    def get_texto_traducir(self):
        return self.texto_traducir

    def set_texto_traducir(self, texto_traducir):
        self.texto_traducir = texto_traducir


class View(tk.Frame):

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.opcion = tk.StringVar()
        idiomas = ("Español", "Aleman", "Portugués", "Ruso", "Coreano","Japones")
        self.combo_idiomas = ttk.Combobox(self.parent,width=10,textvariable=self.opcion,values=idiomas)
        self.combo_idiomas.current(0)
        self.combo_idiomas.grid(column=0, row=4)


        self.txt_lbl()
        self.botones_widget()
        self.configurar_ventana()

    def txt_lbl(self):
        
        def on_entry_validate(S): return S.isalpha()
        vcmd = (root.register(on_entry_validate),'%S')

        self.lblTextoTraducir = tk.Label(self.parent,text="texto a traducir: ").grid(row=0, column=0)

        self.txtTraducir = tk.Entry(self.parent, validate="key", validatecommand=vcmd)
        self.txtTraducir.grid(row=0,column=1,padx=10,pady=10,ipadx=10,ipady=30)

        self.lblTextoTraducido = tk.Label(self.parent,text="texto traducido: ").grid(row=0, column=2)
        self.lblTextoTraducido = tk.Label(self.parent, text="").grid(row=0, column=3)

        self.lblres = tk.Label(self.parent, text="Resultado").grid(row=3, column=0)

    def configurar_ventana(self):
        self.parent.geometry("480x300")
        self.parent.resizable(0, 0)

    def botones_widget(self):
        self.btnguardar = tk.Button(text="Guardar")
        self.btnguardar.grid(row=2, column=0)

        self.btnmostrar = tk.Button(text="Mostrar")
        self.btnmostrar.grid(row=2, column=1)

    def mostrar_resultado(self, message):
        self.lblres['text'] = message

    def mostrar_error(self,message):
        self.lblres['text'] = message

    def campo_vacio (self, message):
        self.lblres['text'] = message

class Controller:

    def __init__(self, root):
        self.model = Model()
        self.view = View(root)

        self.view.btnguardar.config(command=self.guardar_texto)
        self.view.btnmostrar.config(command=self.traducir_el_texto)

    def guardar_texto(self):
        try:
            self.model.texto_traducir = self.view.txtTraducir.get()
            a = self.view.combo_idiomas.get()
            print(a)
        except:
            self.borrar_campos()
            xd = "Verifica que los campos no esten vacío"
            print(xd)
            self.view.campo_vacio(xd)

    def mostrar_texto(self):
        texto = self.model.get_texto_traducir()
        self.view.lblTextoTraducido['text'] = "Texto traducido", texto

    def borrar_campos(self):
        try:
            self.view.txtTraducir.delete(0, tk.END)
        except Exception as a:
            print(a)

    def muestra_traduccion(self):
        texto_traducido = self.model.get_texto_traducir()
        self.view.lblres['text'] = texto_traducido

    def agregar_datos_generales_json(self, n1, n2, res):
        informacion_json_final = []

        if self.existe_historial() == True:
            nuevos_datos = {"numero1": n1, "numero2": n2, "resultado": res}
            with open("historial.json") as archivo_json:
                datos = json.load(archivo_json)
            datos.append(nuevos_datos)

            with open("historial.json", 'w') as archivo_json:
                json.dump(datos, archivo_json, indent=3)
                print("Se han añadido los siguientes datos al archivo " +archivo_json.name + "\n")
        else:

            informacion_usuario = {"numero1": n1,"numero2": n2,"resultado": res}
            with open("historial.json", 'w') as archivo_json:
                informacion_json_final.append(informacion_usuario)
                json.dump(informacion_json_final, archivo_json, indent=3)
                print(archivo_json.name + " creado exitosamente")

    def existe_historial(self):
        try:
            with open('historial.json') as archivo:
                return True
        except FileNotFoundError as e:
            return False

    def combo_seleccion(self):
        if self.view.combo_idiomas.get() == "Español":
            return "spa_Latn"
        if self.view.combo_idiomas.get() == "Aleman":
            return "deu_Latn"
        if self.view.combo_idiomas.get() == "Portugués":
            return "por_Latn"
        if self.view.combo_idiomas.get() == "Ruso":
            return "rus_Cyrl"
        if self.view.combo_idiomas.get() == "Coreano":
            return "kor_Hang"
        if self.view.combo_idiomas.get() == "Japones":
            return "jpn_Jpan"

    def traducir_el_texto(self):
        idioma = self.combo_seleccion()
        client = nlpcloud.Client("nllb-200-3-3b","0c763b98f814c4649754c8c6e50425f99969aa72",gpu=False)
        texto_traducido = client.translation(self.model.get_texto_traducir(),source="eng_Latn",target=idioma)
        self.view.lblres['text'] = texto_traducido


if __name__ == "__main__":



class Vista:
    def __init__(self):
        self.ventana1=tk.Tk()
        self.ventana1.title("Programa ")
        self.ventana1.geometry("500x300")
        self.lbln1=tk.Label(self.ventana1, text="Numero 1")
        self.lbln1.grid( row=0,column=0)
        self.txtn1=tk.Entry()
        self.txtn1.grid(row=0,column=1)

        self.lbln2=tk.Label(self.ventana1, text="Numero 2")
        self.lbln2.grid(row=1,column=0)
        self.txtn2=tk.Entry()
        self.txtn2.grid(row=1,column=1)   

        self.lblres=tk.Label(self.ventana1, text="Resultado")
        self.lblres.grid(row=3,column=0)
        self.btncalcular=tk.Button(text="Calcular",command= self.boton_guardar_clicked)
        self.btncalcular.grid(row=2,column=0)

        self.control = None
        self.ventana1.mainloop()

    def mostrar_resultado(self, message):
        self.lblres['text'] = message

    def set_control(self, control):
        self.control = control

    def boton_guardar_clicked(self):
        if self.control:
            self.control.mostrar_resultado(self.lbln1.get())
 

class Modelo:
    def __init__(self,n1):
        self.numero1 = n1

    def set_numero1(self,n1):
        self.numero1 = n1        

    def get_numero1(self):
        return self.numero1       

class Controlador:

    def __init__(self, modelo, vista):
        self.modelo = modelo
        self.vista = vista

    def muestra_numero (self, n1):
        self.modelo.numero1 = n1
        self.vista.mostrar_resultado(f'El numero es {n1}')

class App:
    def __init__(self):
        super().__init__()
        modelo = Modelo(2)
        vista = Vista()
        control = Controlador(modelo,vista)
        vista.set_control(control)       

if __name__ == '__main__':
    ap = App()
"""  


openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-davinci-002",
  prompt="Translate this into 1. French, 2. Spanish and 3. Japanese:\n\nWhat rooms do you have available?\n\n1. Quels sont les chambres que vous avez disponibles?\n2. ¿Qué habitaciones tienen disponibles?\n3. あなたはどんな部屋を持っていますか？",
  temperature=0.3,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)