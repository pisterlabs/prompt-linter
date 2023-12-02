import discord
from discord import Embed
from discord.ext import commands
import openai
import random
import asyncio

import os
from dotenv import load_dotenv
from utils.log_config import setup_logging, logging
setup_logging()
logger = logging.getLogger(__name__)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

ABBY_CHAT = 1103490012500201632
DAILY_GUST = 802461884091465748

breeze_lounge = "<#802512963519905852>"
welcome_leaf = "<#858231410682101782>"

roles = {
    "Musician": 808129993460023366,
    "Streamer": 1131231727675768953,
    "Gamer": 1131920998350995548,
    "Developer": 1131231948862398625,
    "Artist": 1131703899842154576,
    "NFT Artist": 1131704410393813003,
    "Writer": 1131704091366654094,
    "Z8phyR Fan": 807678887777140786
}

phrases = [
    "Z8phyR here, and I'm really happy to have you here. Please feel free to tag me and chat!",
    "Hey there! Z8phyR here, ready to chat. Don't hesitate to tag me and let's have a great conversation!",
    "Z8phyR reporting for duty! Feel free to tag me and let's dive into a lively chat!",
    "Z8phyR is here and excited to chat. Tag me and let the conversation begin!",
    "I'm Z8phyR and I may be exploring the outdoors! Tag me and let's have a wonderful chat under the virtual sky!",
    "Hello, I am Z8phyR and am all ears! Tag me and let's have a fantastic chat together!",
    "Hey there, it's Z8phyR ready to bring the good vibes! Don't forget to tag me and let's have an awesome chat!",
    "Greetings, I'm Z8phyR and eager to chat with you! Tag me and let's make this conversation memorable!",
    "Hii :), I'm Z8phyR, the music maestro, reporting for chat duty! Tag me and let's harmonize in a lively conversation!",
    "I'm Z8phyR and here to chat! Tag me and let's have a great conversation!",
    "Z8phyR here, extending a warm welcome! Feel free to tag me and let's have a delightful chat!",
    "Hey there! Z8phyR checking in, ready to chat and have a great time. Tag me and let's dive into a conversation!",
    "Hello there! I'm Z8phyR reporting for chat duty! Tag me and let's embark on a lively discussion together!",
    "Great to meet you! I'm Z8phyR here and thrilled to chat. Tag me and let's kickstart this conversation with enthusiasm!",
    "I'm the owner, Z8phyR embracing the virtual world! Tag me and let's engage in a fantastic chat under the digital sky!",
    "Hi - I'm Z8phyR! I'm all ears and excited to chat! Tag me and let's make this conversation truly memorable!",
    "Whats up!!! I'm Z8phyR bringing the positive vibes! Don't forget to tag me and let's have an amazing chat!",
    "Hey there, I'm Z8phyR , here and eager to connect. Tag me and let's create a conversation worth cherishing!",
    "I'm the owner - Z8phyR, the music enthusiast, ready for a chat! Tag me and let's groove to the rhythm of conversation!",
    "I'm Z8phyR - here to chat! Tag me and let's have an enjoyable conversation filled with laughter and insights!"
]

class Welcome(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    # GPT Call
    def generate_message(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200
        )
        return response.choices[0].text.strip()

    @commands.Cog.listener()
    async def on_ready(self):
        logger.info(f"[👋] Starting Welcome Scheduler")

    @commands.Cog.listener()
    async def on_member_join(self, member):
        if member.bot:
            return
        await asyncio.sleep(60) # Wait for 60 sec
        await self.welcome_new_member(member)


    async def welcome_new_member(self, member):
        logger.info("[👋] Welcoming new member!")
        guild = member.guild
        channel = guild.get_channel(DAILY_GUST)

        heart = "<a:z8_leafheart_excited:806057904431693824>"

        prompt = f"""
        I'm Abby, virtual assistant for the Breeze Club Discord, I will generate a welcome message for a new member in the server on the behalf of Z8phyR, our server owner and musician. I'll make sure to include a warm creative welcome, a reminder to check the leaf of rules in {welcome_leaf} channel and introduce themselves at the {breeze_lounge}, and an encouragement to join in the conversations. I won't forget to add some bunny charm! 🐰🥕🌳
        They have selected these roles:
        """

        # Check roles
        role_ids = [role.id for role in member.roles]
        logger.info(f"[👋] Role IDs for new member: {role_ids}")
        user_roles = []

        if roles["Musician"] in role_ids:
            user_roles.append('musician')
        if roles["Streamer"] in role_ids:
            user_roles.append('streamer')
        if roles["Gamer"] in role_ids:
            user_roles.append('gamer')
        if roles["Developer"] in role_ids:
            user_roles.append('developer')
        if roles["Artist"] in role_ids:
            user_roles.append('artist')
        if roles["NFT Artist"] in role_ids:
            user_roles.append('NFT artist')
        if roles["Writer"] in role_ids:
            user_roles.append('writer')
        if roles["Z8phyR Fan"] in role_ids:
            user_roles.append(' a big fan of Z8phyR!')

        if user_roles:
            prompt += ' ' + ' '.join(user_roles)
        logger.info(f"[👋] User roles selected: {user_roles}")
        # logger.info(f"User Roles have been added to prompt: {prompt}")

        message_content = self.generate_message(prompt)
        await channel.send("**Attention <@&807664341158592543>**")
        embed = Embed(
            title=f"{heart} **Welcome our newest member!** {heart}",
            description=f"Kind greetings, {member.mention}! {heart} \n{message_content}",
            color=0x00ff00
        )
        # Add a line above the description
        embed.set_author(name="The Winds brings another to the Breeze Club!")

        # Add a line below the description
        random_phrase = random.choice(phrases)
        embed.set_footer(text=f"🍃 {random_phrase}")

        message = await channel.send(embed=embed)
        await message.add_reaction(heart)

async def setup(bot):
    await bot.add_cog(Welcome(bot))


