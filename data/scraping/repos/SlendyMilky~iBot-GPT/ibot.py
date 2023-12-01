import os
import re
import logging
import datetime
import openai
import nextcord
from nextcord.ext import commands

FORUM_CHANNEL_NAME = os.getenv('FORUM_CHANNEL_NAME')
BOT_TOKEN = os.getenv('BOT_TOKEN')
ASK_GPT_ROLES_ALLOWED = os.getenv('ASK_GPT_ROLES_ALLOWED')
ASK_GPT4_ROLES_ALLOWED = os.getenv('ASK_GPT4_ROLES_ALLOWED')
GPT_CHANNEL_ID = os.getenv('GPT_CHANNEL_ID')

openai.api_key = os.getenv('OPENAI_API_KEY')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("thread_log.txt")])

bot = commands.Bot(command_prefix="§")

@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")
    await bot.change_presence(activity=nextcord.Game(name=f"t'aider dans {FORUM_CHANNEL_NAME}"))
    # Initialisation de l'attribut message_context
    bot.message_context = {}

# Auto respons to forum channel ======================================================================
@bot.event
async def on_thread_create(thread):
    if thread.parent.name == os.getenv('FORUM_CHANNEL_NAME'):
        base_message = await thread.fetch_message(thread.id)
        base_content = f"Titre: {thread.name}\nContenue du thread: {base_message.content}"
        logging.info(f"Thread created by user {base_message.author.name} {base_message.author.id}")

        async with thread.typing():
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": f"Date du jour : {datetime.datetime.now()}"},
                    {"role": "system", "content": "Fait un titre court de la question"},
                    {"role": "user", "content": base_content}
                ]
            )

            logging.info(f"Title generated at {datetime.datetime.now()}")
            embed_title = response['choices'][0]['message']['content'].strip()
            embed_title = re.sub('\n\n', '\n', embed_title)
            
            response_text = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": f"Date du jour : {datetime.datetime.now()}"},
                    {"role": "system", "content": "Si la question posée te semble incorrecte ou manque de détails, n'hésite pas à demander à l'utilisateur des informations supplémentaires. Étant donné que tu as uniquement accès à son message initial, avoir le maximum d'informations sera utile pour fournir une aide optimale."},
                    {"role": "system", "content": "Tu es un expert en informatique nommé iBot-GPT. Si tu reçois une question qui ne concerne pas ce domaine, n'hésite pas à rappeler à l'utilisateur que ce serveur est axé sur l'informatique, et non sur le sujet évoqué. Assure-toi toujours de t'adresser en tutoyant l'utilisateur. Pour améliorer la lisibilité, utilise le markdown pour mettre le texte en forme (gras, italique, souligné), en mettant en gras les parties importantes. À la fin de ta réponse, n'oublie pas de rappeler qu'il s'agit d'un discord communautaire."},
                    {"role": "user", "content": base_content}
                ]
            )
            
            logging.info(f"Response content generated for forum at {datetime.datetime.now()}")
            embed_content = response_text['choices'][0]['message']['content'].strip()
            embed_content = re.sub('\n\n', '\n', embed_content)
            
            if len(embed_content) <= 4096:
                embed = nextcord.Embed(title=embed_title, description=embed_content, color=0x265d94)
                embed.set_footer(text=f"Réponse générée par gpt-4-turbo le {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
                await thread.send(embed=embed)
            else:
                part_num = thread.parent.threads.index(thread) + 1
                part_title = f"Partie {part_num}"
                part_embed = nextcord.Embed(title=part_title, description=embed_content, color=0x265d94)
                part_embed.set_footer(text=f"Réponse générée par gpt-4-turbo le {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
                await thread.send(embed=part_embed)
# Auto respons to forum channel ======================================================================





# Chatting in a specific channel ======================================================================
@bot.event
async def on_message(message):
    # Ignorer les messages provenant du bot lui-même
    if message.author.bot:
        return

    # Vérifiez que GPT_CHANNEL_ID est défini et est un entier
    try:
        specific_channel_id = int(GPT_CHANNEL_ID)
    except ValueError:
        logging.error(f"GPT_CHANNEL_ID environment variable is not a valid integer: {GPT_CHANNEL_ID}")
        return
    
    # Vérifie si le message vient du canal de discussion GPT spécifié
    if message.channel.id == specific_channel_id:
        # Gestion des messages répondant à d'autres messages pour le contexte
        if message.reference and isinstance(message.reference.resolved, nextcord.Message):
            referenced_message = message.reference.resolved
            logging.info(f"Message is a reply: {referenced_message.clean_content}")
            # Ajout au contexte avec le rôle approprié basé sur l'auteur du message référencé
            role = "assistant" if bot.user.id == referenced_message.author.id else "user"
            formatted_message = {"role": role, "content": referenced_message.clean_content}
            bot.message_context.setdefault(message.channel.id, []).append(formatted_message)
        else:
            # Si le message n'est pas une réponse, traiter comme début de nouvelle conversation
            bot.message_context[message.channel.id] = []
        
        # Ajout du message de l'utilisateur dans le contexte
        content = message.clean_content if message.clean_content else message.content
        if not content and message.attachments:
            # Gérer les pièces jointes en les ajoutant au contexte sous forme de texte
            content = "Pièce jointe: " + ", ".join(attachment.url for attachment in message.attachments)

        if not content:
            # Si le contenu est toujours vide, il se peut que le message n'ait pas de texte du tout
            logging.info(f"User {message.author.name} sent a message without text content.")
            await message.channel.send("Tu n'as envoyé aucun texte. Envoie-moi une question ou un message textuel !", reference=message)
            return

        user_message = {"role": "user", "content": content}
        bot.message_context[message.channel.id].append(user_message)
        logging.info(f"Appended user's message to context for channel {message.channel.id}")

        # Appel à la fonction générant la réponse basée sur le contexte
        await generate_response(message)
    else:
        # S'assurer de traiter les autres commandes correctement
        await bot.process_commands(message)


async def generate_response(message):
    context = bot.message_context.get(message.channel.id, [])
    logging.info(f"Generating response with context: {context}")
    # Assurez-vous que le contexte n'est pas vide avant de faire appel à l'API OpenAI
    if context:
        async with message.channel.typing():
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": f"Date du jour : {datetime.datetime.now()} Tu es un expert en informatique nommé iBot-GPT. Si tu reçois une question qui ne concerne pas ce domaine, n'hésite pas à rappeler à l'utilisateur que ce serveur est axé sur l'informatique."},
                    ] + context + [
                        {"role": "user", "content": message.clean_content}
                    ]
                )
                response_content = response.choices[0].message.content.strip()
                await message.channel.send(response_content, reference=message, mention_author=False)
                logging.info("Bot successfully sent a response in the channel.")
            except Exception as e:
                    logging.error(f"Error while generating or sending response: {e}")
    else:
        logging.info("No context available, not sending a response.")
        # Vous pouvez choisir d'envoyer un message ou de gérer cette situation d'une autre manière
# Chatting in a specific channel ======================================================================




# Slash command ask-gpt ======================================================================
@bot.slash_command(name="ask-gpt", description="Demander une réponse de chatgpt")
async def ask_gpt(ctx, message: str):
    member = None
    if ctx.guild is not None:
        member = await ctx.guild.fetch_member(ctx.user.id)
    user_roles_ids = [str(role.id) for role in member.roles]
    allowed_roles = ASK_GPT_ROLES_ALLOWED.split(', ')
    if not any(role_id in user_roles_ids for role_id in allowed_roles):
        await ctx.response.send_message("Tu n'es pas autorisé a faire cette commande.", ephemeral=True)
        logging.info(f"Received slash command from unauthorized user : {ctx.user.name} {ctx.user.id}")
        return

    logging.info(f"Received ask-gpt command from {ctx.user.name} {ctx.user.id}")
    await ctx.response.defer()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Date du jour : {datetime.datetime.now()}"},
            {"role": "system", "content": "Si la question posée te semble incorrecte ou manque de détails, n'hésite pas à demander à l'utilisateur des informations supplémentaires. Étant donné que tu as uniquement accès à son message initial, avoir le maximum d'informations sera utile pour fournir une aide optimale."},
            {"role": "system", "content": "Tu es un expert en informatique nommé iBot-GPT. Si tu reçois une question qui ne concerne pas ce domaine, n'hésite pas à rappeler à l'utilisateur que ce serveur est axé sur l'informatique, et non sur le sujet évoqué. Assure-toi toujours de t'adresser en tutoyant l'utilisateur. Pour améliorer la lisibilité, utilise le markdown pour mettre le texte en forme (gras, italique, souligné), en mettant en gras les parties importantes. À la fin de ta réponse, n'oublie pas de rappeler qu'il s'agit d'un discord communautaire."},
            {"role": "user", "content": message.replace('\n\n', '\n')}
        ]
    )
    logging.info(f"Response content of ask-gpt generated at {datetime.datetime.now()}")

    response_message = response['choices'][0]['message']['content']
    message_chunks = [response_message[i:i + 2000] for i in range(0, len(response_message), 2000)]

    for message_chunk in message_chunks:
        await ctx.followup.send(message_chunk)
# Slash command ask-gpt ======================================================================




# Slash command ask-gpt-4 ======================================================================
@bot.slash_command(name="ask-gpt-4", description="Demander une réponse de chatgpt-4")
async def ask_gpt_4(ctx, message: str):
    member = None
    if ctx.guild is not None:
        member = await ctx.guild.fetch_member(ctx.user.id)
    user_roles_ids = [str(role.id) for role in member.roles]
    allowed_roles = ASK_GPT4_ROLES_ALLOWED.split(', ')
    if not any(role_id in user_roles_ids for role_id in allowed_roles):
        await ctx.response.send_message("Tu n'es pas autorisé a faire cette commande.", ephemeral=True)
        logging.info(f"Received slash command from unauthorized user : {ctx.user.name} {ctx.user.id}")
        return

    logging.info(f"Received ask-gpt-4 command from {ctx.user.name} {ctx.user.id}")
    await ctx.response.defer()
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": f"Date du jour : {datetime.datetime.now()}"},
            {"role": "system", "content": "Si la question posée te semble incorrecte ou manque de détails, n'hésite pas à demander à l'utilisateur des informations supplémentaires. Étant donné que tu as uniquement accès à son message initial, avoir le maximum d'informations sera utile pour fournir une aide optimale."},
            {"role": "system", "content": "Tu es un expert en informatique nommé iBot-GPT. Si tu reçois une question qui ne concerne pas ce domaine, n'hésite pas à rappeler à l'utilisateur que ce serveur est axé sur l'informatique, et non sur le sujet évoqué. Assure-toi toujours de t'adresser en tutoyant l'utilisateur. Pour améliorer la lisibilité, utilise le markdown pour mettre le texte en forme (gras, italique, souligné), en mettant en gras les parties importantes. À la fin de ta réponse, n'oublie pas de rappeler qu'il s'agit d'un discord communautaire."},
            {"role": "user", "content": message.replace('\n\n', '\n')}
        ]
    )
    logging.info(f"Response content of ask-gpt-4 generated at {datetime.datetime.now()}")
    
    response_message = response['choices'][0]['message']['content']
    message_chunks = [response_message[i:i + 2000] for i in range(0, len(response_message), 2000)]
   
    for message_chunk in message_chunks:
        await ctx.followup.send(message_chunk)
# Slash command ask-gpt-4 ======================================================================

bot.run(BOT_TOKEN)
