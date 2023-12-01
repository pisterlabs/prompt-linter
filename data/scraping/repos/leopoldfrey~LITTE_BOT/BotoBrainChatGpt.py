#!/usr/bin/env python3
from pyosc import Client, Server
import os, signal, functools, socket
print = functools.partial(print, end='\n',flush=True)

import openai
# Apply the API Key
openai.api_key = "YOUR_API_KEY_HERE" #https://platform.openai.com/account/api-keys

botophoneIP = "192.168.1.66"

class BotoBrain:
    # initialisation de la classe
    def __init__(self):
        # récupération de l'IP de la machine
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        self.ip = s.getsockname()[0]
        s.close()

        self.osc_server = Server(self.ip, 14001, self.oscIn)
        self.osc_client = Client(botophoneIP, 14000) #"192.168.1.66"
        self.osc_client.send("/ip", self.ip)

        # ICI VOUS POUVEZ INITIALISER VOS VARIABLES ET CHARGER LE MODELE

        print("[BotoBrain] Ready, "+self.ip)

    # initalisation de conversation déclenché par le controleur principal (quand on décroche le téléphone)
    def newConversation(self):
        # MODIFIER ICI POUR REINITIALISER LA CONVERSATION
        pass

    # première phrase du bot, c'est lui qui lance la conversation
    def speakStart(self):
        # MODIFIER LA (ou les) PREMIERE PHRASE
        reponse = "Bonjour"
        self.osc_client.send('/lastresponse',reponse)

    # fin de conversation déclenchée par le controleur principal (temps max ou nombre d'interactions max)
    def endConversation(self, phrase):
        # MODIFIER LA (ou les) PHRASE DE FIN
        reponse = "Au revoir et à bientôt"
        self.osc_client.send('/end',reponse)

    # phrase(s) de relance quand silence trop long
    def relance(self):
        # MODIFIER LA (ou les) PHRASE DE RELANCE
        reponse = "Parlons d'autre chose"
        self.osc_client.send('/lastresponse', reponse)

    # phrase(s) que dit le bot quand il perd l'utilisateur
    def areYouThere(self):
        # MODIFIER LA (ou les) PHRASE DE RELANCE
        reponse = "Êtes-vous toujours là ?"
        self.osc_client.send('/lastresponse', reponse)

    # c'est ici qu'il faut insérer le BOT, string "phrase" en entrée
    def getResponse(self, phrase):
        print("[BotoBrain] user: "+phrase)

        if(phrase == "au revoir"): # EXEMPLE DE MOT-CLE
            # pour arrêter la conversation
            self.osc_client.send('/end',"Vous partez déjà ? Au revoir alors.")
        else:
            # MODIFIER ICI pour traiter la phrase de l'utilisateur "phrase"
            reponse = self.generate_response(phrase)
            print("[BotoBrain] chatgpt: "+reponse)
            self.osc_client.send('/lastresponse', reponse)

    # Generate a response using OpenAI GPT-3
    def generate_response(self, prompt):
        # completions = openai.Completion.create(
        #     engine="text-davinci-002",
        #     prompt=prompt,
        #     max_tokens=200,
        #     n=1,
        #     stop=None,
        #     temperature=0.5,
        # )

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        message = completion['choices'][0]['message']['content']
        return message


# NE PAS MODIFIER SOUS CETTE LIGNE
# NE PAS MODIFIER SOUS CETTE LIGNE
# NE PAS MODIFIER SOUS CETTE LIGNE

    #messages reçus par le contrôleur principal
    def oscIn(self, address, *args):
        print("OSC IN ", address, args[0])
        if(address == '/getresponse'):
            self.getResponse(args[0])
        elif(address == '/newConversation'):
            self.newConversation()
        elif(address == '/relance'):
            self.relance()
        elif(address == '/start'):
            self.speakStart()
        elif(address == '/end'):
            self.endConversation(args[0])
        elif(address == '/areYouThere'):
            self.areYouThere()
        else:
            print("[BotoBrain] OSC IN : "+str(address))
            for x in range(0,len(args)):
                print("     " + str(args[x]))

    # terminaison
    def kill(self):
        self.osc_server.stop()
        os._exit(0)

# gestion de terminaison de processus par signal
def handler(signum, frame):
    boto.kill()

signal.signal(signal.SIGINT, handler)

# MAIN
if __name__ == "__main__":
    boto = BotoBrain()
