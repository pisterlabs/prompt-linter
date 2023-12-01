import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import base64
import requests
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from openai import OpenAI
from playsound import playsound

load_dotenv()
class ChatInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Interface OpenAI")
        # Variables d'instance
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.message_history = [{"role": "system", "content": "L'intégralité de la conversation doit être en français ,ton objectif est d'être un assistant juridique ,la réponse en maximum  4000 token "}]
        self.assistant_message = ""
        self.image_path = ""
        self.pdf_path = ""
        self.texte = ""
        self.client = OpenAI(api_key= self.api_key)
        

        # Configuration de l'interface utilisateur
        self.setup_ui()

    def setup_ui(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.response_text = tk.Text(top_frame, height=35, width=100)
        self.response_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(bottom_frame, text="Selectionne ton PDF", command=self.select_pdf).pack(side=tk.LEFT, padx=5, pady=5)
        self.pdf_label = ttk.Label(bottom_frame, text="Pas de PDF sélectionnée")
        self.pdf_label.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(bottom_frame, text="Image", command=self.select_image).pack(side=tk.LEFT, padx=5, pady=5)
        self.image_label = ttk.Label(bottom_frame, text="Image pas sélectionnée")
        self.image_label.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(bottom_frame, text="Entrez votre Prompt:").pack(side=tk.LEFT, padx=5, pady=5)
        self.prompt_entry = tk.Text(bottom_frame, height=2, width=50)
        self.prompt_entry.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(bottom_frame, text="Envoie la requête", command=self.send_request).pack(side=tk.LEFT, padx=5, pady=5)
       
        self.var = tk.IntVar()
        ttk.Checkbutton(bottom_frame, text="Réponse vocale", command=self.checkbutton_callback, variable=self.var).pack(side=tk.LEFT, padx=5, pady=5)
        print(self.var)

    def checkbutton_callback(self):
        if self.var.get() == 1:
            print("Checkbutton est coché.")
            print(self.var.get())
            
        else:
            print("Checkbutton n'est pas coché.")
            print(self.var.get())

    def play_audio_from_file(self, file_path):
        playsound(file_path)
        
    def select_image(self):
        self.image_path = filedialog.askopenfilename()
        self.image_label.config(text=f"Selectionne l'image: {self.image_path}")

    def select_pdf(self):
        self.pdf_path = filedialog.askopenfilename()
        self.pdf_label.config(text=f"Selectionne un PDF: {self.pdf_path}")
        raw_texte = self.convert_pdf_to_images(self.pdf_path)
        self.texte = self.filter_text(raw_texte)

    def filter_text(self, text):
        caracteres_autorises = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789éèêëàâäîïôöùûüç ')
        filtered_text = ''.join(char for char in text if char in caracteres_autorises)
        return filtered_text
    
        
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def send_request(self):
        user_prompt = self.prompt_entry.get("1.0", tk.END).strip()
        self.prompt_entry.delete("1.0", tk.END)
        # Ne rien faire si le prompt est vide
        if not user_prompt:
            return  
        #Information d'entête pour l'API
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        #Contenu du message 
        user_message = {"role": "user", "content": user_prompt}
        #ajout du contenu à notre requéte final 
        self.message_history.append(user_message)
        #ajout de l'historique à notre requéte 
        self.message_history.append({"role": "assistant", "content": self.texte})
        #choix du modéle si il y a une image ou non 
        if self.image_path:
            model="gpt-4-vision-preview"
        else:
            model="gpt-4-1106-preview"        
        #réquete final dont  message qui comprend: systéme (les informations global sur le comportement),assistant qui comprend le pdf si il y en a un et User qui comprend le message de l'utilisateur 
        payload = {"model": model, "messages": self.message_history.copy(), "max_tokens": 4000}
        #modifier le contenu de user_message si il y a une image 
        if self.image_path:
            base64_image = self.encode_image(self.image_path)
            user_message["content"] = [{"type": "text", "text": user_prompt},
                                       {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
        #requete et récupération de la sortie 
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()
        new_response = response_json["choices"][0].get("message", {}).get("content", "")
        if new_response !=None and self.var.get() == 1:
            audio_response = self.client.audio.speech.create(
                model="tts-1",
                voice="onyx",
                input=new_response
            )
            # Enregistrez l'audio dans un fichier temporaire
            # Créez un fichier temporaire dans le répertoire courant
            temp_file_name = "temp_audio.mp3"
            
            # Enregistrez l'audio dans le fichier temporaire
            audio_response.stream_to_file(temp_file_name)

            # Jouez l'audio
            self.play_audio_from_file(temp_file_name)

            # Supprimez le fichier temporaire après que la lecture soit terminée
            os.remove(temp_file_name)
            
        #Affichage et stockage du fil de discussion pour que l'historique soit visible et pris en compte
        self.assistant_message += "\nVous: " + user_prompt + "\nAssistant: " + new_response
        self.response_text.configure(state=tk.NORMAL)
        self.response_text.insert(tk.END, "\nVous: " + user_prompt + "\nAssistant: " + new_response)
        self.response_text.configure(state=tk.DISABLED)
        #ajout de la réponse à notre historique pour qu'on puisse l'utiliser dans la requête suivante 
        self.message_history.append({"role": "assistant", "content": new_response})

        #on vide le path  image et  pdf 
        self.image_path = ""
        self.image_label.config(text="Image non selectionné")
        self.pdf_path = ""
        self.pdf_label.config(text="Image non selectionné")

    def convert_pdf_to_images(self, pdf_path):
        document = fitz.open(pdf_path)
        texte = ""
        for page in document:
            texte += page.get_text()
        return texte
    
# Utilisation de la classe
root = tk.Tk()
#ajout du théme
root.tk.call('source', 'azure.tcl')
root.tk.call('set_theme', 'dark')  
app = ChatInterface(root)
root.mainloop()