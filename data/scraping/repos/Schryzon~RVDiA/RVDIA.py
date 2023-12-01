"""
Schryzon/Jayananda (11)
G-Tech Re'sman Programming Division
RVDiA (Revolutionary Virtual Discord Assistant)
Feel free to modify and do other stuff.
Contributions are welcome.
Licensed under the MIT LICENSE.
* Note: Now that RVDiA is verified, I own the rights to the name.
        Making public clones of her under the same name is a big no no, okay sweetie?
"""

import asyncio
import discord
import os
import openai
import aiohttp
from time import time
from dotenv import load_dotenv
from pkgutil import iter_modules
from scripts.help_menu.help import Help
from cogs.General import Regenerate_Answer_Button
from discord.ext import commands, tasks
from random import choice as rand
from contextlib import suppress
from datetime import datetime
from scripts.main import connectdb, titlecase, check_vote
load_dotenv('./secrets.env') # Loads the .env file from python-dotenv pack

class RVDIA(commands.AutoShardedBot):
  """
  A subclass of commands.AutoShardedBot; RVDiA herself.
  This is in order to make her attributes easier to maintain.
  (Nah, I'm just lazy tbh.)
  """
  def __init__(self, **kwargs):
    self.synced = False
    self.__version__ = "v1.1.6"
    self.event_mode = False
    self.color = 0x86273d
    self.runtime = time() # UNIX float
    self.coin_emoji = "<:rvdia_coin:1121004598962954300>"
    self.coin_emoji_anim = "<a:rvdia_coin_anim:1121004592033955860>"
    self.rvdia_emoji = '<:rvdia:1140812479883128862>'
    self.rvdia_emoji_happy = '<:rvdia_happy:1121412270220660803>'
    self.cyron_emoji = '<:cyron:1082789553263349851>' # Join up!!!

    super().__init__(
      command_prefix=commands.when_mentioned, 
      case_insensitive=True, 
      strip_after_prefix=False, 
      intents=discord.Intents.default(), # Finally got to this stage.

      help_command=Help(
            no_category = "Tak tergolongkan", 
            color = self.color,
            active_time = 60,
            image_url = os.getenv('bannerhelp') if not self.event_mode else os.getenv('bannerevent'),
            index_title = "Kategori Command",
            timeout=20,
            case_insensitive = True
        ),
      **kwargs
    )


rvdia = RVDIA() # Must create instance

cogs_list = [cogs.name for cogs in iter_modules(['cogs'], prefix='cogs.')] # iter_modules() for easier task

@rvdia.event
async def on_connect():
    print("RVDiA has connected.")

@rvdia.event
async def on_ready():
    """
    Detect when RVDiA is ready (not connected to Discord).
    """
    await rvdia.wait_until_ready() # So I "don't" get rate limited
    for cog in cogs_list:
      if not cog == 'cogs.__init__':
          await rvdia.load_extension(cog)
    print('Internal cogs loaded!')
    
    if not rvdia.synced:
      synced_commands = await rvdia.tree.sync() # Global slash commands sync, also returns a list of commands.
      await asyncio.sleep(1.5) # Avoid rate limit
      await rvdia.tree.sync(guild=discord.Object(997500206511833128)) # Wonder if it fixes with this??
      rvdia.synced = [True, len(synced_commands)]
      print('Slash Commands up for syncing!')

    if not change_status.is_running():
      change_status.start()
      print('Change status starting!')

    update_guild_status.start()

    print("RVDiA is ready.")


@tasks.loop(minutes=5)
async def change_status():
  """
  Looping status, rate = 5 minute
  """
  is_event = 'Event mode ON!' if rvdia.event_mode == True else 'Standard mode'
  users = 0
  for guilds in rvdia.guilds:
    users += guilds.member_count -1
  user_count_status = f'{users} users'
  all_status=['in my room', 'in G-Tech Server', '"How to be cute"', 'you', 'G-Tech members',
                  'Ephotech 2023', user_count_status, f'{rvdia.__version__}',
                  '/help', 'in my dream world', 'Add me!', is_event, '~♪',
                  'Re:Volution'
                ]
  status = rand(all_status)
  # Just count, I'm trying to save space!
  if status == all_status[2] or status == all_status[4] or status == user_count_status:
    type = discord.Activity(type=discord.ActivityType.watching, name=status)
  elif status == all_status[3]:
    type = discord.Activity(type=discord.ActivityType.listening, name=status)
  elif status == all_status[6]:
    type = discord.Activity(name = status, type = 5)
  else:
    type = discord.Game(status)
  await rvdia.change_presence(status = discord.Status.idle, activity=type)


@tasks.loop(hours=1)
async def update_guild_status():
    """
    Sends data regarding shard and server count to Top.gg
    """
    try:
      headers = {'Authorization': os.getenv('topggtoken')}
      async with aiohttp.ClientSession(headers=headers) as session:
          await session.post(f'https://top.gg/api/bots/{rvdia.user.id}/stats', data={
              'server_count':len(rvdia.guilds),
              'shard_count':rvdia.shard_count
          })
          print(f'Posted server updates to Top.gg!')

    except Exception as error:
       print(f'Error sending server count update!\n{error.__class__.__name__}: {error}')

@rvdia.command(aliases = ['on', 'enable'], hidden=True)
@commands.is_owner()
async def load(ctx, ext):
  """
  Manually load cogs
  """
  if ext == "__init__":
    await ctx.send(f"Stupid.")
    return
  try:
    rvdia.load_extension(f"cogs.{ext}")
    await ctx.send(f"Cog `{ext}.py` sekarang aktif!")
  except commands.ExtensionAlreadyLoaded:
    await ctx.send(f"Cog `{ext}.py` sudah diaktifkan!")
  except commands.ExtensionNotFound:
    await ctx.send(f"Cog `{ext}.py` tidak ditemukan!")


@rvdia.command(aliases = ['off', 'disable'], hidden=True)
@commands.is_owner()
async def unload(ctx, ext):
  """
  Manually unload cogs
  """
  if ext == "__init__":
    await ctx.send(f"Stupid.")
    return
  try:
    rvdia.unload_extension(f"cogs.{ext}")
    await ctx.send(f"Cog `{ext}.py` sekarang tidak aktif!")
  except commands.ExtensionNotFound:
    await ctx.send(f"Cog `{ext}.py` tidak ditemukan!")
  except commands.ExtensionNotLoaded:
    await ctx.send(f"Cog `{ext}.py` sudah dimatikan!")

@rvdia.command(hidden = True)
@commands.is_owner()
async def cogs(ctx):
    """
    Cogs list
    """
    embed = discord.Embed(title = "RVDIA Cog List", description = "\n".join(cogs_list), color = ctx.author.colour)
    embed.set_thumbnail(url = rvdia.user.avatar)
    embed.set_footer(text = "Cogs were taken from \".RVDIA/cogs\"")
    await ctx.send(embed=embed)

@rvdia.command(hidden=True)
@commands.is_owner()
async def refresh(ctx):
  """
  In case something went horribly wrong
  """
  with suppress(commands.ExtensionNotLoaded):
    for cog in cogs_list:
      if not cog == 'cogs.__init__':
          await rvdia.unload_extension(cog)
          await rvdia.load_extension(cog)
  await ctx.reply('Cogs refreshed.')

@rvdia.command(hidden=True)
@commands.is_owner()
async def restart(ctx:commands.Context): # In case for timeout
   await ctx.send('Restarting...')
   print('!!RESTART DETECTED!!')
   await rvdia.close()
   await asyncio.sleep(2)
   rvdia.run(token=os.getenv('token'))
   await rvdia.wait_until_ready()
   await ctx.channel.send('RVDIA has restarted!')

@rvdia.command(hidden=True)
@commands.is_owner()
async def status(ctx:commands.Context, *, status):
   if status.lower() == 'restart' or status.lower() == 'reset':
      if not change_status.is_running:
         return change_status.start()
   change_status.cancel()
   await rvdia.change_presence(status = discord.Status.idle, activity=discord.Game(status))
   await ctx.reply('Changed my status!')

@rvdia.command(hidden=True)
@commands.is_owner()
async def blacklist(ctx:commands.Context, user:discord.User, *, reason:str=None):
   match user.id:
      case rvdia.owner_id:
         return await ctx.reply('Tidak bisa blacklist owner!')
      case rvdia.user.id:
         return await ctx.reply('Tidak bisa blacklist diriku sendiri!')
      case _:
         pass
      
   blacklisted = await connectdb('Blacklist')
   check_blacklist = await blacklisted.find_one({'_id':user.id})
   if not check_blacklist:
      await blacklisted.insert_one({'_id':user.id, 'reason':reason})
      embed = discord.Embed(title='‼️ BLACKLISTED ‼️', timestamp=ctx.message.created_at, color=0xff0000)
      embed.description = f'**`{user}`** telah diblacklist dari menggunakan RVDIA!'
      embed.set_thumbnail(url=user.avatar.url if not user.avatar is None else os.getenv('normalpfp'))
      embed.add_field(name='Alasan:', value=reason, inline=False)
      return await ctx.reply(embed=embed)
   
   await ctx.reply(f'`{user}` telah diblacklist!')

@rvdia.command(hidden=True)
@commands.is_owner()
async def whitelist(ctx:commands.Context, user:discord.User):
   blacklisted = await connectdb('Blacklist')
   check_blacklist = await blacklisted.find_one({'_id':user.id})
   if not check_blacklist:
      return await ctx.reply(f'**`{user}`** tidak diblacklist dari menggunakan RVDIA!')
   
   await blacklisted.find_one_and_delete({'_id':user.id})
   await ctx.reply(f'`{user}` telah diwhitelist!')

@rvdia.event
async def on_message(msg:discord.Message):
    """
    Replacing the available on_message event from Discord
    TO DO: Create check_blacklist() and only run it here.
    Configure RVDiA's class
    """

    if not msg.guild:
        return
    
    await rvdia.process_commands(msg) # Execute commands from here

    if msg.author.bot == True:
        return

    if msg.reference:
        try:
          fetched_message = await msg.channel.fetch_message(msg.reference.message_id)
          match fetched_message.author.id:
              case rvdia.user.id:
                  pass
              case _:
                  return
          
          if fetched_message.embeds and fetched_message.embeds[0] and fetched_message.embeds[0].footer:
              message_embed = fetched_message.embeds[0]
          else:
              return
          
          if message_embed.footer.text == 'Jika ada yang ingin ditanyakan, bisa langsung direply!':    
            async with msg.channel.typing():
              embed_desc = message_embed.description
              embed_title = message_embed.title
              author = message_embed.author.name
              openai.api_key = os.getenv('openaikey')
              message = msg.content
              currentTime = datetime.now()
              date = currentTime.strftime("%d/%m/%Y")
              hour = currentTime.strftime("%H:%M:%S")
              result = await openai.ChatCompletion.acreate(
                  model="gpt-3.5-turbo",
                  temperature=1.2,
                  messages=[
                  {"role":'system', 'content':os.getenv('rolesys')+f' You are currently talking to {msg.author}'},
                  {"role":"assistant", 'content':f'The current date is {date} at {hour} UTC+8 | {author} said: {embed_title} | Your response was: {embed_desc}'},
                  {"role": "user", "content": message}
                  ]
              )

              if len(message) > 256:
                message = message[:253] + '...' #Adding ... from 253rd character, ignoring other characters.

              embed = discord.Embed(
                title=' '.join((titlecase(word) for word in message.split(' '))), 
                color=msg.author.color, 
                timestamp=msg.created_at
                )
              embed.description = result['choices'][0]['message']['content']
              embed.set_author(name=msg.author)
              embed.set_footer(text='Jika ada yang ingin ditanyakan, bisa langsung direply!')
              regenerate_button = Regenerate_Answer_Button(message)
              await msg.channel.send(embed=embed, view=regenerate_button)
            return
          
          elif message_embed.footer.text == 'Reply \"Approve\" jika disetujui\nReply \"Decline\" jika tidak disetujui':
            old_acc_field = message_embed.fields[0].value
            old_acc_string = old_acc_field.split(': ')
            old_acc_id = int(old_acc_string[2].strip())

            new_acc_field = message_embed.fields[1].value
            new_acc_string = new_acc_field.split(': ')
            new_acc_id = int(new_acc_string[2].strip())
            user = await rvdia.fetch_user(new_acc_id)

            database = await connectdb('Game')
            if msg.content.lower() == "approve" or msg.content.lower() == "accept":
                old_data = await database.find_one({'_id':old_acc_id})
                keep = {
                    'level':old_data['level'],
                    'exp':old_data['exp'],
                    'next_exp':old_data['next_exp'],
                    'last_login':old_data['last_login'],
                    'coins':old_data['coins'],
                    'karma':old_data['karma'],             
                    'attack':old_data['attack'],
                    'defense':old_data['defense'],
                    'agility':old_data['agility'],
                    'special_skills':old_data['special_skills'],    
                    'items':old_data['items'],
                    'equipments':old_data['equipments']
                }

                await database.find_one_and_update({'_id':new_acc_id}, {'$set':keep})
                await database.delete_one({'_id':old_acc_id})
                await msg.channel.send(f'✅ Transfer akun untuk {user} selesai!')
                try:
                   await user.send(f"✅ Request transfer akun Re:Volution-mu telah selesai!\nApproved by: `{msg.author}`")
                except:
                   return
                
            elif msg.content.lower() == "decline" or msg.content.lower() == "deny":
              await fetched_message.delete()
              await msg.channel.send(f"❌ Request transfer akun untuk {user} tidak disetujui")
              try:
                  await user.send(f"❌ Mohon maaf, request transfer data akun Re:Volutionmu tidak disetujui.\nUntuk informasi lebih lanjut, silahkan hubungi `{msg.author}` di https://discord.gg/QqWCnk6zxw")
              except:
                  return
        
        except Exception as e:
           if "currently overloaded with other requests." in str(e):
              return await msg.channel.send('Maaf, fitur ini sedang dalam gangguan. Mohon dicoba nanti!')
           elif "overloaded or not ready" in str(e) or "Bad gateway." in str(e):
              return await msg.channel.send("Sepertinya ada yang bermasalah dengan otakku tadi.\nTolong coba ulangi pertanyaanmu lagi!")
           elif "rate limit reached" in str(e).lower():
              return await msg.channel.send("Aduh, maaf, otakku sedang kepanasan.\nTolong tanyakan lagi setelah 20 detik!")
           elif "unknown message" in str(e).lower() or 'message_id: Value "None" is not snowflake.' in str(e) or "404 not found" in str(e).lower() or "Invalid Form Body In message_reference: Unknown message" in str(e):
              return await msg.channel.send("Hah?!\nSepertinya aku sedang mengalami masalah menemukan pesan yang kamu reply!")
           elif "403 Forbidden" in str(e) or "Missing Access" in str(e):
              try:
                 return await msg.channel.send("Aku kekurangan `permission` untuk menjalankan fitur ini!\nPastikan aku bisa mengirim pesan dan embed di channel ini!")
              except:
                 try:
                    return await msg.author.send("Aku kekurangan `permission` untuk menjalankan fitur ini!\nPastikan aku bisa mengirim pesan dan embed di channel itu!")
                 except:
                    return
           await msg.channel.send('Ada yang bermasalah dengan fitur ini, aku sudah mengirimkan laporan ke developer!')
           channel = rvdia.get_channel(906123251997089792)
           await channel.send(f'`{e}` Untuk fitur balasan GPT-3.5 Turbo!')
           print(e)

# Didn't know I'd use this, but pretty coolio
if __name__ == "__main__":
  rvdia.run(token=os.getenv('token'))