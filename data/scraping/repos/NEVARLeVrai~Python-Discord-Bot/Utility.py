import discord
from discord.ext import commands
import random
import io
import asyncio
from cogs import Help
import traceback
import openai
import datetime
import typing

class utility(commands.Cog):
    def __init__(self, client):
        self.client = client
        self.reponse_en_cours = False  # Variable de verrouillage initialement à False
        with open("C:/Users/danie/Mon Drive/tokengpt.txt", "r") as f:
            GPT_API_KEY = f.read().strip()
        openai.api_key = GPT_API_KEY
        self.rate_limit_delay = 1  # Délai en secondes entre chaque requête (1 seconde dans cet exemple)

        
    def is_bot_dm(message):
        return message.author.bot and isinstance(message.channel, discord.DMChannel)
    
    def is_bot_dm(self, message):
        return message.author == self.client.user and isinstance(message.channel, discord.DMChannel)

    async def send_tts(self, vc, lang, vol, text):
        # Découpe le texte en parties de longueur maximale max_length
        max_length = 200
        text_parts = [text[i:i+max_length] for i in range(0, len(text), max_length)]

        # Joue chaque partie du texte
        for part in text_parts:
            vc.play(discord.FFmpegPCMAudio(
                executable="C:/ProgramData/chocolatey/lib/ffmpeg-full/tools/ffmpeg/bin/ffmpeg.exe",
                source=f"http://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl={lang}&q={part}",
                options=f"-af volume={vol}"
            ))
            while vc.is_playing():
                await asyncio.sleep(1)

    @commands.Cog.listener()
    async def on_ready(self):
        print("Utility.py is ready")

    @commands.command()
    async def tts(self, ctx, lang="fr", vol="3.0", *, text):
        await ctx.message.delete()
        vc = None
        try:
            if ctx.author.voice:
                vc = await ctx.author.voice.channel.connect()
            else:
                embed5 = discord.Embed(title="TTS Play", description="Vous devez être dans un salon vocal pour utiliser cette commande.", color=discord.Color.red())
                embed5.set_author(name=f"Demandé par {ctx.author.name}", icon_url=ctx.author.avatar)
                embed5.set_footer(text=Help.version1)
                await ctx.send(embed=embed5, delete_after=5)
                return

            embed1 = discord.Embed(title="TTS Play", color=discord.Color.green())
            embed1.set_author(name=f"Demandé par {ctx.author.name}", icon_url=ctx.author.avatar)
            embed1.add_field(name="Volume:", value=f"**{vol}**")
            embed1.add_field(name="Langue:", value=f"**{lang}**")
            embed1.add_field(name="Dit:", value=f"**{text}**", inline=False)
            embed1.set_footer(text=Help.version1)
            await ctx.send(embed=embed1, delete_after=25)
            await self.send_tts(vc, lang, vol, text)

        except Exception as e:
            traceback_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            embed_error = discord.Embed(title="TTS Play - Erreur", description=f"Une erreur s'est produite lors de la lecture TTS:\n\n```\n{traceback_str}\n```", color=discord.Color.red())
            embed_error.set_author(name=f"Demandé par {ctx.author.name}", icon_url=ctx.author.avatar)
            embed_error.set_footer(text=Help.version1)
            await ctx.send(embed=embed_error, delete_after=10)
        finally:
            if vc:
                await vc.disconnect()
                embed3 = discord.Embed(title="TTS Play", description="Déconnecté avec succès!", color=discord.Color.green())
                embed3.set_author(name=f"Demandé par {ctx.author.name}", icon_url=ctx.author.avatar)
                embed3.set_footer(text=Help.version1)
                await ctx.send(embed=embed3, delete_after=5)

    @commands.command(aliases=["dm"])
    @commands.has_permissions(administrator=True)
    async def mp(self, ctx, user: discord.User, *, message: str = None):
        await ctx.message.delete()

        files = ctx.message.attachments
        if files:
            for file in files:
                await user.send(file=await file.to_file())

        if message:
            await user.send(message)

        # Message d'information
        embed = discord.Embed(title="Message Privé Envoyé!", description=f"Message envoyé à **{user.name}#{user.discriminator}**", color=discord.Color.gold())
        embed.set_author(name=f"Demandé par {ctx.author.name}", icon_url=ctx.author.avatar)
        embed.set_footer(text=Help.version1)

        await ctx.send(embed=embed, delete_after=10)


        
        

    @commands.command(aliases=["repeat"])
    async def say(self, ctx, destination: typing.Union[discord.TextChannel, discord.Member, str], *, message=None):
        await ctx.message.delete()

        if isinstance(destination, str):
            if destination.startswith("<#") and destination.endswith(">"):
                channel_id = int(destination[2:-1])  # Extraction de l'ID à partir de la mention
                destination = self.client.get_channel(channel_id)
                if not isinstance(destination, discord.TextChannel):
                    await ctx.send("Salon invalide spécifié.")
                    return
            else:
                embed1 = discord.Embed(title="Message Non Envoyé!", description="Format de mention de salon incorrect.", color=discord.Color.red())
                embed1.set_author(name=f"Demandé par {ctx.author.name}", icon_url=ctx.author.avatar)
                embed1.set_footer(text=Help.version1)

                await ctx.send(embed=embed1, delete_after=10)
                return

        # Vérifiez si des fichiers sont attachés au message
        if ctx.message.attachments:
            files = [await attachment.to_file() for attachment in ctx.message.attachments]
        else:
            files = []

        await destination.send(message, files=files)

        # Ajouter un message d'information
        embed = discord.Embed(title="Message Envoyé!", description=f"Message envoyé à {destination.mention}", color=discord.Color.green())
        embed.set_author(name=f"Demandé par {ctx.author.name}", icon_url=ctx.author.avatar)
        embed.set_footer(text=Help.version1)
        await ctx.send(embed=embed, delete_after=10)





        

    @commands.has_permissions(administrator=True)
    @commands.command(aliases=["deldms"])
    async def delmp(self, ctx):
        await ctx.message.delete()
        try:
            total_deleted = 0

            # Envoie d'un message global indiquant que la suppression des DMs est en cours
            embed = discord.Embed(title="Suppression des messages privés en cours...", color=discord.Color.yellow())
            embed.set_footer(text=Help.version1)
            status_msg = await ctx.send(embed=embed, delete_after=10)

            tasks = []
            for member in ctx.guild.members:
                if not member.bot:
                    dm_channel = await member.create_dm()
                    messages_to_delete = [msg async for msg in dm_channel.history() if self.is_bot_dm(msg)]
                    deleted_count = len(messages_to_delete)

                    if deleted_count > 0:
                        tasks.append(dm_channel.send(f"Suppression Terminé!", delete_after=10))
                        tasks.append(asyncio.gather(*[msg.delete() for msg in messages_to_delete]))
                        await asyncio.sleep(self.rate_limit_delay)  # Limite de taux

                    total_deleted += deleted_count

                    # Envoyer un message individuel pour chaque utilisateur dont les DMs ont été supprimés
                    if deleted_count > 0:
                        embed = discord.Embed(title=f"Messages privés de **{member.name}#{member.discriminator}** supprimés !", color=discord.Color.green())
                        embed.add_field(name="Nombre de messages supprimés", value=str(deleted_count))
                        embed.set_footer(text=Help.version1)
                        tasks.append(ctx.send(embed=embed, delete_after=10))
                        await asyncio.sleep(self.rate_limit_delay)  # Limite de taux
                    


            # Attendre que toutes les tâches soient terminées
            await asyncio.gather(*tasks)
            
            if total_deleted > 0:
                embed1 = discord.Embed(title=f"Messages privés supprimés au total.", description=f"{total_deleted}", color=discord.Color.purple())
            else:
                embed1 = discord.Embed(title="Aucun message privé à supprimer.", color=discord.Color.red())
            embed1.set_footer(text=Help.version1)
            await ctx.send(embed=embed1, delete_after=10)
            
        except Exception as e:
            traceback.print_exc()


    @commands.command(aliases=["8ball"])
    async def magicball(self, ctx, * ,question):
        await ctx.message.delete()
        responses=['Comme je le vois oui.',
                  'Oui.',
                  'Positif',
                  'De mon point de vue, oui',
                  'Convaincu.',
                  'Le plus probable.',
                  'De grandes chances',
                  'Non.',
                  'Négatif.',
                  'Pas convaincu.',
                  'Peut-être.',
                  'Pas certain',
                  'Peut-être',
                  'Je ne peux pas prédire maintenant.',
                  'Je suis trop paresseux pour prédire.',
                  'Je suis fatigué. *continue à dormir*']
        response = random.choice(responses)
        embed=discord.Embed(title="La Boule Magique 8 à parlé!", color=discord.Color.purple())
        embed.set_author(name=f"Demandé par {ctx.author.name}", icon_url=ctx.author.avatar)
        embed.add_field(name='Question: ', value=f'{question}')
        embed.add_field(name='Réponse: ', value=f'{response}')
        embed.set_footer(text=Help.version1)
        with open("./Autres/8ball.png", "rb") as f:
            image_data = f.read()
        embed.set_thumbnail(url="attachment://8ball.png")
        await ctx.send(embed=embed, file=discord.File(io.BytesIO(image_data), "8ball.png"))
        
    @commands.command()
    async def hilaire(self, ctx):
        await ctx.message.delete()
        responses = ["le protocole RS232",
                "FTTH",
                "Bit de Start",
                "Bit de parité",
                "Sinusoïdale",
                "RJ45",
                "Trop dbruiiiit!!!!",
                "Raphaël les écouteurs",
                "Can le téléphone",
                "JoOoAnnY",
                "Le théorème de demorgan"]
        responses = random.choice(responses)
        embed=discord.Embed(title="Wiliam Hilaire à parlé!", color=discord.Color.purple())
        embed.set_author(name=f"Demandé par {ctx.author.name}", icon_url=ctx.author.avatar)
        embed.add_field(name='Hilaire à dit: ', value=f'{responses}')
        embed.set_footer(text=Help.version1)
        with open("./Autres/hilaire.png", "rb") as f:
            image_data = f.read()
        embed.set_thumbnail(url="attachment://hilaire.png")
        await ctx.send(embed=embed, file=discord.File(io.BytesIO(image_data), "hilaire.png"))
        

    @commands.command()
    async def gpt(self, ctx, *, question):
        if self.reponse_en_cours:
            await ctx.send("\nUne réponse est déjà en cours de génération. Veuillez patienter.", delete_after=5)
            if isinstance(ctx.channel, discord.TextChannel):
                await ctx.message.delete()
            return

        self.reponse_en_cours = True  # Définir le verrouillage sur True

        try:
            async with ctx.typing():
                response = self.gpt_reponse(question)
                response = self.nettoyer_texte(response)
                response_with_mention = f"{ctx.author.mention}\n{response}"  # Ajouter la mention à la réponse
            await ctx.send(response_with_mention)

            with open("C:/Users/danie/Mon Drive/gptlogs.txt", "a") as f:
                current_time = datetime.datetime.now()
                f.write(f"Date: {current_time.strftime('%Y-%m-%d')}\n")
                f.write(f"Heure: {current_time.strftime('%H:%M:%S')}\n")
                f.write(f"User: {ctx.author.mention}\n")                
                f.write(f"Question: {question}\n")
                f.write(f"Réponse: {response}\n")
                f.write("-" * 50 + "\n")

        finally:
            self.reponse_en_cours = False  # Réinitialiser le verrouillage à False

    def gpt_reponse(self, question):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=question,
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7
        )
        bot_response = response.choices[0].text.strip()
        print("\n\nChat GPT:")
        print(f"Question: {question}")
        print(f"Réponse: {bot_response}")
        return bot_response
        

    def nettoyer_texte(self, texte):
        # Supprimer les sauts de ligne redondants
        texte_nettoye = "\n".join(line for line in texte.splitlines() if line.strip())
        return texte_nettoye



    @commands.command()
    async def dalle(self, ctx, *, question):
        if self.reponse_en_cours:
            await ctx.send("\nUne réponse est déjà en cours de génération. Veuillez patienter.", delete_after=5)
            if isinstance(ctx.channel, discord.TextChannel):
                await ctx.message.delete()
            return

        self.reponse_en_cours = True  # Définir le verrouillage sur True

        try:
            async with ctx.typing():
                response = self.dalle_reponse(question)
                response_with_mention = f"{ctx.author.mention}\n{response}"  # Ajouter la mention à la réponse
            await ctx.send(response_with_mention)

            with open("C:/Users/danie/Mon Drive/dallelogs.txt", "a") as f:
                current_time = datetime.datetime.now()
                f.write(f"Date: {current_time.strftime('%Y-%m-%d')}\n")
                f.write(f"Heure: {current_time.strftime('%H:%M:%S')}\n")
                f.write(f"User: {ctx.author.mention}\n")                
                f.write(f"Question: {question}\n")
                f.write(f"Réponse: {response}\n")
                f.write("-" * 50 + "\n")

        finally:
            self.reponse_en_cours = False  # Réinitialiser le verrouillage à False

    def dalle_reponse(self, question):
        response = openai.Image.create(
        prompt=question,
        n=1,
        size="1024x1024"
        )
        bot_response = response['data'][0]['url']
        print("\n\nDall-E:")
        print(f"Question: {question}")
        print(f"Réponse: {bot_response}")
        return bot_response
        

async def setup(client):
    await client.add_cog(utility(client))