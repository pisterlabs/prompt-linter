# Discord BOT with openio
# Dr CADIC Philippe / @fulfuroid
# Si les librairies discord et openai ne sont pas installees
# Tapez:
#           pip install discord
#           pip install openai

# Utilisation des librairies Python 3
import discord
import os
import openai

client = discord.Client()  # Connexion a discord

openai.api_key = "Ajouter votre clé openai ici" # clé d'identification avec OPenAI

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')   # evenement de connexion au serveur discord

@client.event
async def on_message(message):   # Si un message arrive, alors on lance son aspiration et on l'envoi dans le moteur d'IA openAI
    print(message.content)

    if message.author == client.user:   #ici cette ligne permet d'éviter les boucles perturbant le robot
        return

    if message.content == "Ping":       # Simmple test: si l'humain tapes 'Ping' alors on répond sans passer pas l'IA
        await message.channel.send("Pong")
    if message.content == "Hello":      # Si l'usager dit 'hello' alors le robot se présente
        await message.channel.send(file=discord.File('charts.png'))
        await message.channel.send("Hello, I'm Sulfuroid's OPEN AI GPT3 engine to reply to your questions...")
    else:                               # Si c'est autre chose, alors on passe le contenu du message à l'IA et on attend sa réponse
        response = openai.Completion.create(model="text-davinci-002", prompt=message.content, temperature=0, max_tokens=255)
        response2 = response['choices'][0]['text'].strip()
        await message.channel.send(response2)


#Client and token
client.run("Ajouter le token du robot discord ici")
