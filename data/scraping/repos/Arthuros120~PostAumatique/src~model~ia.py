# -*- coding: utf-8 -*-
"""Ia class"""

import logging
import openai


class Ia:

    def __init__(self, config):

        self.logger = logging.getLogger('PostAumatique-Log')

        self.logger.info("Initialisation de l'IA...")
        self.logger.info("Chargement de la configuration...")
        self.config = config

        self.logger.info("Chargement de l'API key...")
        self.api_key = self.config.get("Ia", "apiKey")

    def generate_text(self, body_path: str, society) -> str:

        self.logger.info("Démarage de la génération de texte...")

        body_text = ""
        speech_text = ""

        self.logger.info("Chargement du fichier de base...")
        with open(body_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:
            body_text += line

        self.logger.info(
            "Chargement du speech pour la lettre de motivation...")

        with open("res/motivationLetter/speech.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:
            speech_text += line

        speech_text = speech_text.replace("¤Society¤", society.get_name())
        speech_text = speech_text.replace("¤body¤", body_text)

        self.logger.info("Attribution de l'API key...")

        openai.api_key = self.api_key

        self.logger.info("Je génère le texte pour toi...")

        if self.config.get("Ia", "safeMode"):

            self.logger.info(
                "Mode safe activé, je génère 3 textes différents pour toi, tu pourras choisir celui que tu préfères.")

            choices = []

            for i in range(0, 3):

                self.logger.info("Génération du texte {} ...".format(i + 1))
                
                while True:
                    
                    try:
                    
                        response = openai.Completion.create(
                            model="text-davinci-003",
                            prompt=speech_text,
                            temperature=0.5,
                            max_tokens=200,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0.6,
                            stop=["¤"]
                        )
                        
                        break
                
                    except Exception as e:
                        
                        self.logger.error("Une erreur est survenue lors de la génération du texte, on recommence...")
                        self.logger.error(e)
                        
                        continue

                choices.append(response.choices[0].text.replace("\n", "")) # type: ignore

            self.logger.info(
                "Voici les 3 propositions de textes que j'ai pu générer pour toi:")

            for i in range(0, len(choices)):
                print()
                self.logger.info("{}: {}".format((i + 1), choices[i]))
                print()

            self.logger.debug("Demande de sélection du texte...")
            select = input("Quel texte veux-tu utiliser ? (1, 2 ou 3) : ")

            self.logger.debug("Retour de l'utilisateur : " + select)

            while select != "1" and select != "2" and select != "3":

                self.logger.debug(
                    "L'utilisateur a entré une valeur incorrecte, on lui redemande...")
                select = input("Quel texte veux-tu utiliser ? (1, 2 ou 3) : ")

                self.logger.debug("Retour de l'utilisateur : " + select)

            self.logger.info("Je choisis le texte {} ...".format(select))

            choice = choices[int(select) - 1]

        else:

            self.logger.warning(
                "Mode safe désactivé, je génère un seul texte pour toi...")

            while True:
                    
                    try:
                    
                        response = openai.Completion.create(
                            model="text-davinci-003",
                            prompt=speech_text,
                            temperature=0.5,
                            max_tokens=200,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0.6,
                            stop=["¤"]
                        )
                        
                        break
                
                    except Exception as e:
                        
                        self.logger.error("Une erreur est survenue lors de la génération du texte, on recommence...")
                        self.logger.error(e)
                        
                        continue

            choice = response.choices[0].text.replace("\n", "") # type: ignore

        self.logger.info("J'ai fini de générer le texte pour toi !")

        return choice
