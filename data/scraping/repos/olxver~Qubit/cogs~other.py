import fileinput
import string
import nextcord as discord
import os
import sqlite3 as sql
import datetime
import random
import aiohttp
import io
import aiofiles
import openai
import asyncio
import contextlib

from bot import start_time
from nextcord.ext import commands
from nextcord.ext.commands import cooldown, BucketType
from nextcord import SlashOption, Interaction
from nextcord.ext import application_checks
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env file

github_token = os.getenv("GITHUB_API_TOKEN")  
token = os.getenv("BOT_TOKEN")
openai_key = os.getenv("OPENAI_API_KEY")




class Other(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    

    @discord.slash_command(description="Get the bot's latency")
    async def ping(self, interaction: Interaction):
        await interaction.send(f"Pong! 🏓 `{round(self.bot.latency * 1000)}ms`", ephemeral=True)
    
    @discord.slash_command(description="Get the bot's uptime")
    async def uptime(self, interaction: Interaction):
        await interaction.send(f"Uptime: <t:{start_time.timestamp():.0f}:R>", ephemeral=True)
    
    @discord.slash_command(name="privacy-policy", description="Privacy policy")
    @commands.cooldown(1, 10, BucketType.user)
    async def privacy_policy(self, interaction:Interaction):
        message = """
            **Introduction**
            This Privacy Policy explains how we collect, use, and protect the information that we gather from users of our Discord bot (Qubit). By using Qubit, you consent to the terms of this Privacy Policy.

            **Information We Collect**
            We collect certain information from you in order for Qubit to function properly, such as your Discord User ID and server information. We may also collect additional information about your usage of Qubit, such as the commands that you use and how frequently you use them.

            **How we use your Information**
            We use the information that we collect from you solely for the purpose of providing Qubit's services. We may use the information to improve Qubit's functionality, to analyze usage trends, or to respond to user inquiries. We may also use the information to send you occasional updates or announcements related to Qubit, but we will not sell or share your information with any third parties without your knowledge and consent.

            **Data Retention**
            We will retain your information for as long as necessary to provide Qubit's services to you. If you would like us to delete your information, please contact olxver#9999. We will delete your information within a reasonable timeframe after receiving your request.

            **Data Security**
            We take reasonable measures to protect the information that we collect from you against unauthorized access, disclosure, or destruction. However, no data transmission over the internet or other network can be guaranteed to be 100% secure. Therefore, we cannot guarantee the security of your information.

            **Children's Privacy**
            Qubit is not intended for use by children under the age of 13. We do not knowingly collect information from children under the age of 13.

            **Links to Third-Party Sites**
            Qubit may contain links to third-party sites or services that are not owned or controlled by us. We are not responsible for the privacy practices of these sites or services.

            **Changes to Privacy Policy**
            We reserve the right to modify this Privacy Policy at any time without prior notice. Your continued use of Qubit after any changes to the Privacy Policy will constitute your acceptance of such changes.

            **Your Rights**
            You have the right to access, correct, update, or delete your information that we have collected from you. You may also have the right to object to or restrict our processing of your information. To exercise any of these rights, please contact olxver#9999.

            **Contact Us**
            If you have any questions or concerns about this Privacy Policy or our privacy practices, please contact olxver#9999.
        """
        embed = discord.Embed(title="Privacy Policy for Qubit", description=message, color=discord.Colour.og_blurple())
        await interaction.send(embed=embed, ephemeral=True)

    @discord.slash_command(name="tos", description="Terms of Service")
    @commands.cooldown(1, 10, BucketType.user)
    async def tos(self, interaction:Interaction):
        message = """
            **Acceptance of Terms**
            By using this Discord bot (Qubit), you agree to be bound by these Terms of Service (the "TOS"). If you do not agree to these terms, please do not use Qubit.

            **Use of Bot**
            Qubit is provided to you for entertainment purposes only. You agree to comply with all applicable laws and regulations when using Qubit.

            **Privacy Policy**
            Qubit collects certain information from you in order to function properly, such as your Discord User ID and server information. This information is used solely for the purpose of providing Qubit's services and will not be shared with any third parties. By using Qubit, you agree to the terms of Qubit's Privacy Policy (see more at /privacy-policy).

            **Disclaimer of Warranties**
            Qubit is provided on an "as is" and "as available" basis, without any warranties of any kind, express or implied. Qubit may be unavailable from time to time due to maintenance or other reasons. We make no warranties or representations about the accuracy or completeness of the content provided by Qubit.

            **Limitation of Liability**
            In no event shall we be liable for any damages whatsoever, including but not limited to, direct, indirect, special, incidental, or consequential damages, arising out of or in connection with the use or inability to use Qubit, even if we have been advised of the possibility of such damages.

            **Changes to TOS**
            We reserve the right to modify these terms at any time without prior notice. Your continued use of Qubit after any changes to the TOS will constitute your acceptance of such changes.

            **Termination**
            We reserve the right to terminate your access to Qubit at any time, without notice, for any reason whatsoever.

            **Governing Law**
            These terms and your use of Qubit shall be governed by and construed in accordance with the laws of the jurisdiction in which Qubit is operated.

            **Severability**
            If any provision of these terms is found to be invalid or unenforceable, the remaining provisions shall remain in full force and effect.

            **Entire Agreement**
            These terms constitute the entire agreement between you and us with respect to Qubit and supersede all prior or contemporaneous communications and proposals, whether oral or written, between you and us.
        """
        embed = discord.Embed(title="Terms of Service for Qubit", description=message, color=discord.Colour.og_blurple())
        await interaction.send(embed=embed, ephemeral=True)

    



def setup(bot):
    bot.add_cog(Other(bot))
    print("Other cog loaded")