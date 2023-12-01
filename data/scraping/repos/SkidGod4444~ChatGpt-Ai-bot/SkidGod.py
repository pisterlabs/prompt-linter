import discord
from discord.ext import commands
import openai

def generate_text(prompt):
  openai.api_key = "put your api here"
  completions = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
  )

  message = completions.choices[0].text
  return message

class skid(commands.Cog):
    def __init__(self, client):
        self.client = client

    @commands.command()
    async def ai(self, ctx, *, text):
      lund = generate_text(text)
      fof =f"**__OpenAi's response:__**\n\n```{lund}```"
      await ctx.reply(fof, ephemeral=True, mention_author=True) 
    @commands.command()
    async def src(self, ctx):
      Skid =f"***Must give a star:*** https://github.com/SkidGod4444/ChatGpt-Ai-bot"
      await ctx.send(Skid)
    @commands.command()
    async def help(self, ctx):
      Skid =f"**OpenAi is here**\n```s!h -> To get my help cmds info\ns!ai -> To search something using the Ai\ns!jsk -> Jhisaku\ns!src -> Get my source code for free!```"
      await ctx.send(Skid)
      #await client.add_cog(skid(client))
async def setup(client): await client.add_cog(skid(client))