from openai import AsyncOpenAI
from . import config
from typing import List
import discord
import asyncio
from .stasilogging import log, log_user, lid
from copy import deepcopy



aclient = AsyncOpenAI(api_key=config.C["openai"]["key"])

max_errors = 6 # 1 and a half minutes

async def make_chatgpt_request(messages: List[dict]):
    res = await aclient.chat.completions.create(model="gpt-3.5-turbo",
    messages=messages)
    return res.choices[0].message

async def make_vetting_chatgpt_request(messages: List[dict]):
    res = await aclient.chat.completions.create(model=config.C["openai"]["vettingmodel"],
    messages=messages)
    return res.choices[0].message

async def make_chatgpt4_request(messages: List[dict]):
    res = await aclient.chat.completions.create(model="gpt-4",
    messages=messages)
    return res["choices"][0]["message"]

def build_verification_embed(user, messages, verdict):
    messages_ = deepcopy(messages)
    if len(messages_) > 25:
        messages_ = messages_[:25]
    embed = discord.Embed(title=f"Verdict: {verdict}", description="Vetting completed." if verdict != "yanked" else "Vetting in progress.")
    if user:
        embed.set_author(name=user, icon_url=user.avatar.url if user.avatar else "https://cdn.discordapp.com/embed/avatars/4.png")
    if verdict == "bgtprb":
        embed.set_footer(text="User is being overtly offensive, exercise caution.")
    elif verdict == "yanked":
        embed.set_footer(text="Interview still in progress.")
    # fill out embed, i
    for message in messages_:
        if len(message["content"]) > 1024:
            message["content"] = message["content"][:1021] + "..."
        embed.add_field(name=message["role"], value=message["content"], inline=False)
    return embed

def build_paginated_verification_embeds(user, messages, verdict):
    messages_ = deepcopy(messages)
    embeds = []
    embed = discord.Embed(title=f"Verdict: {verdict}", description="Vetting completed." if verdict != "yanked" else "Vetting in progress.")
    if user:
        embed.set_author(name=user, icon_url=user.avatar.url if user.avatar else "https://cdn.discordapp.com/embed/avatars/4.png")
    if verdict == "bgtprb":
        embed.set_footer(text="User is being overtly offensive, exercise caution.")
    elif verdict == "yanked":
        embed.set_footer(text="Interview still in progress.")
    embeds.append(embed)
    # fill out embed, i
    for i, message in enumerate(messages_):
        embed = discord.Embed(title=f"Message {i+1} of {len(messages_)}", description="Vetting completed." if verdict != "yanked" else "Vetting in progress.")
        if len(message["content"]) < 1024:
            embed.add_field(name=message["role"], value=message["content"], inline=False)
        elif len(message["content"]) < 2048:
            embed.add_field(name=message["role"], value=message["content"][:1024], inline=False)
            embed.add_field(name=message["role"], value=message["content"][1024:], inline=False)
        elif len(message["content"]) < 3072:
            embed.add_field(name=message["role"], value=message["content"][:1024], inline=False)
            embed.add_field(name=message["role"], value=message["content"][1024:2048], inline=False)
            embed.add_field(name=message["role"], value=message["content"][2048:], inline=False)
        elif len(message["content"]) < 4096:  # this should be the highest under any circumstances
            embed.add_field(name=message["role"], value=message["content"][:1024], inline=False)
            embed.add_field(name=message["role"], value=message["content"][1024:2048], inline=False)
            embed.add_field(name=message["role"], value=message["content"][2048:3072], inline=False)
            embed.add_field(name=message["role"], value=message["content"][3072:], inline=False)

        if user:
            embed.set_author(name=user, icon_url=user.avatar.url if user.avatar else None)
        embeds.append(embed)
    return embeds

class VettingInterviewer:

    user: None
    errors_in_a_row = 0

    async def generate_response(self):
        global errors_in_a_row
        try:
            response = await make_vetting_chatgpt_request(self.messages)
            self.errors_in_a_row = 0
            self.messages.append({"role": "assistant", "content": ["content"]})
            return response.content
        
        except Exception as e:
            self.errors_in_a_row += 1
            if self.errors_in_a_row == 1:
                await self.user.send("🚫 There has been an error. Please wait.")
            
            if self.errors_in_a_row > max_errors:
                content = f"The web request has failed {max_errors} in a row. [CLOSE:LEFT]"
                self.messages.append({"role": "assistant", "content": content})
                return content
            else:
                await asyncio.sleep(15)
                return await self.generate_response()

    def verdict_check(self, message):
        message = message.upper()
        if "LEFT]" in message:
            return "left"
        elif "RIGHT]" in message:
            return "right"
        elif "REDO]" in message:
            return "redo"
        elif "BGTPRB]" in message:
            return "bgtprb"
        else:
            return False

    async def vet_user(self, ctx: discord.ApplicationContext, user):

        log("aivetting", "startvetting", f"{lid(self)} starting vetting for {log_user(user)}")

        self.user = user
        try:
            dm_channel = await ctx.author.create_dm()
            with dm_channel.typing(): 
                await self.generate_response()
                await dm_channel.send(self.messages[-1]["content"])
        except discord.Forbidden:
            log("aivetting", "dmfail", f"Failed to send DM to {log_user(user)}")
            await ctx.respond("I cannot send messages to you. Please enable DMs from server members and try again.", ephemeral=True)
            return "redo"
    
        await ctx.respond("Check your DMs", ephemeral=True)
        
        while not self.verdict_check(self.messages[-1]["content"]) and len(self.messages) < 32:
            try:
                message = await ctx.bot.wait_for("message", check=lambda m: m.author == user and not m.guild, timeout=60*20)  # 20 minutes to answer
                self.messages.append({"role": "user", "content": message.clean_content})
                log("aivetting", "readmessage", f"{lid(self)} Read message from {log_user(user)}: {message.clean_content}")
            except asyncio.TimeoutError:
                log("aivetting", "timeout", f"User {log_user(user)} timed out.")
                await user.send("SYSTEM: You have timed out (20 minutes). Please try again later.")
                return "redo"
        
            with dm_channel.typing():
                response = await self.generate_response()
                await dm_channel.send(response)
        
        verdict = self.verdict_check(self.messages[-1]["content"])

        return verdict 

    def __init__(self):
        log("aivetting", "newmod", f"New interviewer {lid(self)} created.")
        self.messages = [{"role": "system", "content": config.C["openai"]["vetting_prompt"]},
            {"role": "user", "content": "[START VETTING]"}].copy()
        self.vetting = True

    def __del__(self):
        log("aivetting", "delmod", f"Interviewer {lid(self)} deleted.")
        

async def tutor_question(question):
    res = await make_chatgpt4_request([{"role": "system", "content": config.C["openai"]["tutor_prompt"]}, {"role": "user", "content": question}])
    return res.content


class BetaVettingInterviewer(VettingInterviewer):

    async def openai_request(self, messages):
        try:
            ret = await make_vetting_chatgpt_request(messages)
            self.errors_in_a_row = 0
            return ret
        
        except Exception as e:
            self.errors_in_a_row += 1
            if self.errors_in_a_row == 1:
                await self.user.send("🚫 There has been an error. Please wait.")

            if self.errors_in_a_row > max_errors:
                content = f"The web request has failed {max_errors} in a row. [CLOSE:LEFT]"
                self.messages.append({"role": "assistant", "content": content})
                return {"role": "assistant", "content": content}
            else:
                log("aivetting", "openaierror", f"Error {lid(e)} {self.errors_in_a_row}/{max_errors} in OpenAI request: {e} for interviewer {id(self)}. Retrying in 15 seconds.")
                await asyncio.sleep(15)
                log("aivetting", "openaierror", f"Error {lid(e)} retrying now.")
                return await self.openai_request(messages)

    async def one_off_assistant(self, system, user):
        messages = [{"role": "system", "content": system}, {"role": "assistant", "content": user}]
        res = await self.openai_request(messages)
        return res.content

    async def one_off(self, system, user):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        res = await self.openai_request(messages)
        return res.content

    async def generate_response(self):
        log("aivetting", "betaairesponse", f"Beta AI {lid(self)} Generating response...")
        response = await self.openai_request(self.messages)
        response = response.content

        if not self.verdict_check(response):  # if there is no resolution code, check if the ai just forgot to include one
            prompt = "Evaluate the message given. If it seems like the user has made a final decision regarding someone else's ideology, respond with \"[END]\" If the user is not sure yet, or the interview is otherwise ongoing, do not respond with the code."
            one_off_response = await self.one_off(prompt, response)
            if "[end]" in one_off_response.lower():
                log("aivetting", "aicorrection", f"AI {lid(self)} forgot to include a resolution code. Message: {response} Prompting it to try again. (User: {log_user(self.user)}")
                self.messages.append({"role": "system", "content": "Now end the interview, remember to include a resolution code in your message."})
                return await self.generate_response()

        self.messages.append({"role": "assistant", "content": response})
        log("aivetting", "betaairesponse", f"Beta AI {lid(self)} Generated response: {response}")
        return response
