from ventana_principal import *
from Modelos.ventana_tres_Modelo import *
from Vistas.ventana_tres_Vista import *
import json
import openai

class Ventana_Tres_Controller:

    def __init__(self, root):
        self.model = Ventana_Tres_Model()
        self.view = Ventana_Tres_View(root)
        self.view.btnEnviar_texto_hacia_IA.config(command=self.chatBot)

    def chatBot(self):
        try:
            openai.api_key =("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")
            conversation ="Fui creado por OpenAI. ¿Cómo te puedo ayudar hoy?"
            #self.view.historial_de_conversacion.insert(END,conversation)
            pregunta_usuario = self.view.txtEntrada_texto_usuario.get()
        
            self.view.historial_de_conversacion.insert(END,"Humano: "+pregunta_usuario)
            if pregunta_usuario == "Adios":
                print("AI: ¡Adiós!")
                self.view.parent.destroy()
            
            conversation += "\nHuman:" + pregunta_usuario + "\nAI:"
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

            respuesta_ia = response.choices[0].text.strip()
            conversation += respuesta_ia

            self.view.historial_de_conversacion.insert(END,"IA: "+respuesta_ia)
            self.view.txtEntrada_texto_usuario.delete(0,END)
            
        except Exception as e:
            print(e)
