import os
import csv
import sys
import discord
intents = discord.Intents.all()
intents.typing = False
intents.presences = False
import enchant
import re
import ssl
from OpenSSL import crypto
import requests
import socket
from io import BytesIO
import json
import aiohttp
import asyncio
import subprocess
from langdetect import detect
from discord.ext import commands
import random
import datetime
from datetime import datetime
from datetime import date
from datetime import time
from datetime import timedelta
from aiocfscrape import CloudflareScraper
import openai

# start an instance of the completion engine
# completion_engine = openai.Completion()

# discord and API tokens need to be environment variables named as below
openai.api_key = os.getenv('OPENAI_API_KEY')
token = os.getenv('DISCORD_TOKEN')
tenorapi = os.getenv('TENOR_API_KEY')
weatherapi = os.getenv('WEATHER_API_KEY')
googleapi = os.getenv('GOOGLE_API_KEY')
reourl = os.getenv('REOURL')
reourl2 = os.getenv('REOURL2')
description = '''To seek and annoy'''
giphy_api_key = os.getenv('GIPHY_API_KEY')

client = commands.Bot(command_prefix='.', description=description, intents=intents)
dictionary = enchant.Dict("en_US")

#allowed users for repeat command
allowed_ids = [340495492377083905, 181093076960411648]

@client.event
async def on_ready():
    await client.change_presence(activity=discord.Game(name="Raccoon"))
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')

# random line function for word matching
def random_line(fname):
    lines = open(fname).read().splitlines()
    return random.choice(lines)



# check if seed is set
# if not that means word of the day (wotd) isn't set so run the exception code
try: seed
except:
    wordslines = 0
    # open the words file and count the lines
    file = open(os.path.join(sys.path[0], 'words.txt'), "r")
    for line in file:
        line = line.strip("\n")
        wordslines += 1
    file.close()
    # set seed to today's date for consistency
    seed = date.today().isoformat().replace('-', '')
    # set yesterday's seed to to yesterday's date
    ydayseed = (date.today() - timedelta(days=1)).isoformat().replace('-', '')
    # set today's seed and get today's line number
    random.seed(seed)
    wotdlineno = random.randrange(1, wordslines)
    # set yesterday's seed and get yesterday's line number
    random.seed(ydayseed)
    ydwotdlineno = random.randrange(1, wordslines)
    # open the file and get the 2 words
    f=open(os.path.join(sys.path[0], 'words.txt'))
    alllines=f.readlines()
    swotd = str.strip(alllines[int(wotdlineno)])
    sywotd = str.strip(alllines[int(ydwotdlineno)])
    # after wotd and ywotd are defined reset the seed and close the file
    random.seed()
    file.close()

# set reactions to word of the day
async def wotdreact(message):
    wotd_emojis = [
    "👌",
    "😂",
    "🔥",
    "😱"
    ]
    for emoji in wotd_emojis:
        await message.add_reaction(emoji)

# set reactions to nice
async def nicereact(message):
    nice_emoji = "<:69:844757057437302856>"
    await message.add_reaction(nice_emoji)

# function that does what it says - also works for times spanning midnight
async def is_time_between(begin_time, end_time, check_time=None):
    # If check time is not given, default to current time
    check_time = check_time or datetime.now().time()
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else: # crosses midnight
        return check_time >= begin_time or check_time <= end_time

async def gpt_response(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content


@client.command(name="gpt", help="Ask GPT-3.5 Turbo a question or send a message")
async def gpt(ctx, *, prompt: str):
    response = await gpt_response(prompt)
    await ctx.send(response)

@client.command()
async def movie(ctx, movie_name):
    with open('movies.csv', 'r') as file:
        reader = csv.reader(file)
        found_movies = []
        for row in reader:
            if row[0].lower().startswith(movie_name.lower()):
                found_movies.append(row)

        if found_movies:
            embed = discord.Embed(title='Search Results', color=discord.Color.blue())
            for movie in found_movies:
                embed.add_field(name='Movie Name', value=movie[0], inline=False)
                embed.add_field(name='View Date', value=movie[1], inline=True)
                embed.add_field(name='RT', value=movie[2], inline=True)
                embed.add_field(name='RTA', value=movie[3], inline=True)
                embed.add_field(name='Fumble Expectation', value=movie[4], inline=True)
                embed.add_field(name='Fumble Execution', value=movie[5], inline=True)
                embed.add_field(name='Fumble Entertainment', value=movie[6], inline=True)
                embed.add_field(name='Fumble Recommendation', value=movie[7], inline=True)
                embed.add_field(name='Hylas Expectation', value=movie[8], inline=True)
                embed.add_field(name='Hylas Execution', value=movie[9], inline=True)
                embed.add_field(name='Hylas Entertainment', value=movie[10], inline=True)
                embed.add_field(name='Hylas Recommendation', value=movie[11], inline=True)
                embed.add_field(name='Enfuego Recommendation', value=movie[12], inline=True)
                embed.add_field(name='illusion Expectation', value=movie[13], inline=True)
                embed.add_field(name='illusion Execution', value=movie[14], inline=True)
                embed.add_field(name='illusion Entertainment', value=movie[15], inline=True)
                embed.add_field(name='illusion Recommendation', value=movie[16], inline=True)
                embed.add_field(name='guil Recommendation', value=movie[17], inline=True)

            await ctx.send(embed=embed)
        else:
            await ctx.send('No matching movies found.')
            
# clap!👏
@client.command(
    pass_context=True,
    help="Put a clap emoji between every word.",
    brief="I will clap for you."
)
async def clap(ctx, *, claptext):
    clapemoji = '👏'
    clapupper = str(claptext).upper().split()
    clapjoin = clapemoji.join(clapupper)
    await ctx.send(str(clapjoin) + clapemoji)

# spongebob text into alternating case SaRcAsM
@client.command(
    pass_context=True,
    help="Changes text provided into alternating case spongebob sarcastic memery.",
    brief="Spongebob your text."
)
async def sb(ctx, *, sbtext):
    sbemoji = "<:sb:755535426118484138>"
    res = ""
    for idx in range(len(sbtext)):
        if not idx % 2:
            res = res + sbtext[idx].upper()
        else:
            res = res + sbtext[idx].lower()
    await ctx.send(sbemoji)
    await ctx.send(str(res))

# why not both?
@client.command(
    pass_context=True,
    help="Combines the awesome of sb and clap.",
    brief="Why not both?"
)
async def sbclap(ctx, *, sbclaptext):
    res = ""
    for idx in range(len(sbclaptext)):
        if not idx % 2:
            res = res + sbclaptext[idx].upper()
        else:
            res = res + sbclaptext[idx].lower()
    sbemoji = "<:sb:755535426118484138>"
    clapemoji = '👏'
    clapsplit = str(res).split()
    clapjoin = clapemoji.join(clapsplit)
    await ctx.send(sbemoji)
    await ctx.send(str(clapjoin) + clapemoji)

# inspire me
@client.command(
    pass_context=True,
    help="Generates an image from inspirobot.",
    brief="Inpsire me."
)
async def inspire(ctx):
    embed = discord.Embed(colour=discord.Colour.blue())
    response = await cs_page('https://inspirobot.me/api?generate=true')
    embed.set_image(url=response)
    await ctx.send(embed=embed)

# get today's word
@client.command(
    pass_context=True,
    help="This will return today's word of the day between 8pm and 12pm eastern.",
    brief="Get today's word."
)
async def wotd(ctx):
    if await is_time_between(time(20, 00), time(23, 59)):
        await ctx.send("The word of the day today was: **" + swotd + "**")
    else:
        await ctx.send("We don't talk about the word of the day until after 8pm eastern.")

# get yesterday's word
@client.command(
    pass_context=True,
    help="This will return yesterday's word of the day.",
    brief="Get yesterday's word."
)
async def ywotd(ctx):
    await ctx.send("Yesterday's word of the day was: **" + sywotd + "**")

@client.command()
async def driveway(ctx):

    # Download the image data using requests
    response = requests.get(reourl)
    image_data = response.content

    # Wrap the image data in a BytesIO object to create a file-like object
    file = BytesIO(image_data)
    file.name = "image.jpg"

    # Create an embedded message with the file
    embed = discord.Embed(title="A Driveway", description="Here's a picture of a driveway:")
    embed.set_image(url="attachment://image.jpg")

    # Send the message with the embedded file
    await ctx.send(file=discord.File(file), embed=embed)

@client.command()
async def backyard(ctx):

    # Download the image data using requests
    response = requests.get(reourl2, verify=False)
    image_data = response.content

    # Wrap the image data in a BytesIO object to create a file-like object
    file = BytesIO(image_data)
    file.name = "image.jpg"

    # Create an embedded message with the file
    embed = discord.Embed(title="A Back Yard", description="Here's a picture of a back yard:")
    embed.set_image(url="attachment://image.jpg")

    # Send the message with the embedded file
    await ctx.send(file=discord.File(file), embed=embed)

@client.command(name='sgif')
async def search_gif(ctx, *args):
    # Parse the user's input to extract the search query
    search_query = ' '.join(args)

    # Retrieve the most popular GIF from the Giphy API based on the search query
    response = requests.get(f'http://api.giphy.com/v1/gifs/search?q={search_query}&api_key={giphy_api_key}&limit=1&rating=g&sort=popularity')
    gif_url = response.json()['data'][0]['images']['original']['url']

    # Create an embed with the GIF and send it to the Discord channel
    embed = discord.Embed()
    embed.set_image(url=gif_url)
    await ctx.send(embed=embed)

@client.command(name='ping')
async def ping_host(ctx, host):
    try:
        # Run the ping command and capture the output
        process = subprocess.Popen(['ping', '-c', '3', host], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        # Extract the round-trip time from the output
        output_str = output.decode('utf-8')
        rtts = [float(x.split('=')[-1].split(' ')[0]) for x in output_str.split('\n') if 'time=' in x]

        # Calculate the average round-trip time and send it to the Discord channel
        avg_rtt = sum(rtts) / len(rtts)
        await ctx.send(f'Average round-trip time to {host}: {avg_rtt:.2f} ms')

    except Exception as e:
        await ctx.send(f'Error: {e}')

@client.command(
    pass_context=True,
    help="This will query the google maps api to get the latitude and longitude. Using that it will query the OpenWeather API for local conditions.",
    brief="Weather for the location specified."
)
async def weather(ctx, *, search):
    session = aiohttp.ClientSession()
    session2 = aiohttp.ClientSession()
    search.replace(' ', '+')
    # get our latitude and longitude
    georesp = await session.get('https://maps.googleapis.com/maps/api/geocode/json?key=' + googleapi + '&address=' + search, ssl=False)
    geodata = json.loads(await georesp.text())
    geolat = str(geodata['results'][0]['geometry']['location']['lat'])
    geolon = str(geodata['results'][0]['geometry']['location']['lng'])
    # pass the lat and lon to the weather api
    weatheresp = await session2.get('http://api.openweathermap.org/data/2.5/weather?appid=' + weatherapi + '&lat=' + geolat + '&lon=' + geolon + '&units=imperial')
    weatherdata = json.loads(await weatheresp.text())
    # create the map url for the embed
    locmapurl = 'https://maps.googleapis.com/maps/api/staticmap?center=' + \
        geolat + ',' + geolon + '&zoom=11&size=600x300&key=' + googleapi
    temp = str(weatherdata['main']['temp'])
    city = weatherdata['name']
    country = weatherdata['sys']['country']
    cond = weatherdata['weather'][0]['description']
    feelslike = str(weatherdata['main']['feels_like'])
    humidity = str(weatherdata['main']['humidity'])
    wind_speed = str(weatherdata['wind']['speed'])
    # populate the embed with returned values
    embed = discord.Embed(title="Weather for " + city + " " + country, colour=discord.Colour.blue())
    embed.set_image(url=locmapurl)
    embed.add_field(name="Current Temp (Feels Like)", value=temp + "F" + " " + "(" + feelslike + "F)")
    embed.add_field(name="Conditions", value=cond)
    embed.add_field(name="Wind Speed", value=wind_speed + " mph")
    embed.add_field(name="Humidity", value=humidity + "%")
    await session.close()
    await session2.close()
    await ctx.send(embed=embed)

@client.command(name="wtf")
async def wtf_command(ctx, member: discord.Member):
    # get the previous message in the channel from the specified member
    async for msg in ctx.channel.history(limit=2):
        if msg.id != ctx.message.id and msg.author == member and not msg.author.bot and not msg.content.startswith('.'):
            text = msg.content
            # remove punctuation and split the message into words
            words = [word.strip('.,?!') for word in text.split()]
            # check each word for spelling errors
            misspelled = [word for word in words if not dictionary.check(word)]
            if len(misspelled) > 0:
                # if misspellings were found, correct them and respond with a corrected message
                corrected = [dictionary.suggest(word)[0] if not dictionary.check(word) else word for word in words]
                corrected_msg = "I think {} meant to say: \"{}\"".format(member.mention, " ".join(corrected))
                await ctx.send(corrected_msg)
            return

@client.command()
async def ip(ctx):
    ip = requests.get('https://api.ipify.org').text
    await ctx.send(f"My public IP: {ip}")

@client.command()
async def sslexpiry(ctx, url: str):
    try:
        cert = ssl.get_server_certificate((url, 443))
        x509 = crypto.load_certificate(crypto.FILETYPE_PEM, cert)
        not_after = x509.get_notAfter().decode('ascii')
        expiry_date = datetime.strptime(not_after, '%Y%m%d%H%M%SZ')
        await ctx.send(f'The SSL certificate for {url} will expire on {expiry_date}.')
    except ssl.SSLError:
        await ctx.send(f'Error: Unable to retrieve SSL certificate for {url}.')
    except socket.gaierror:
        await ctx.send(f'Error: Unable to connect to {url}. Please check the URL and try again.')

@client.command()
async def repeat(ctx, channel_mention, *, message):
    if ctx.author.id not in allowed_ids:
        await ctx.send("You are not allowed to use this command.")
        return

    channel_id = re.findall(r'\d+', channel_mention)[0]
    channel = client.get_channel(int(channel_id))
    await channel.send(message)

# match various patterns including word of the day and respond with
# either random lines or react to wotd with emoji
@client.event
async def on_message(message):
    nicepattern = r".*\bnice\b\W*"
    namestr = "marcus"
    moviestr = "movie night"
    herzogstr = "herzog"
    herstr = "amanda"
    kartstr = "kart"
    mariostr = "mario game"
    ff2str = "ff2"
    typongstr = "typong"
    ffstr = "final fantasy"
    neatostr = "neato"
    zeldastr = "zelda"
    titwstr = "this is the way"
    bofhpattern = r".*\berror\b\W*"
    daystr = "what a day"

    # this is the *real* bot username - not the nickname
    botstr = client.user.name

    # what channel is this? needed for channel.send
    channel = message.channel

    if message.author == client.user:
        return
    
#    if namestr.lower() in message.content.lower():
#        response = await gpt_response(message.content)
#        await message.channel.send(response)

    # If the message is in Russian, send "our robot, comrade" in Russian
    lang = detect(message.content)
    if lang == 'ru':
        await message.channel.send('Наш робот, товарищ')


    # reacts to namestr if not botstr
    if (namestr.lower() in message.content.lower()) and (botstr.lower() not in message.content.lower()):
        await channel.send(random_line(os.path.join(sys.path[0], 'name.txt')))
    if botstr.lower() in message.content.lower():
        line = random_line(os.path.join(sys.path[0], 'botmention.txt'))
        response = line.replace("BOT", botstr)
        await channel.send(response)
    
    # wotd reaction
    if swotd.lower() in message.content.lower():
        await wotdreact(message)

    # nice reaction
    sequence = message.content.lower()
    if re.match(nicepattern, sequence):
        await nicereact(message)

    # bofh regex
    if re.match(bofhpattern, sequence):
        await channel.send(random_line(os.path.join(sys.path[0], 'bofh.txt')))

    # other word matches with static and random line responses below
    if moviestr.lower() in message.content.lower():
        await channel.send(random_line(os.path.join(sys.path[0], 'movienight.txt')))

    if herzogstr.lower() in message.content.lower():
        await channel.send(random_line(os.path.join(sys.path[0], 'herzog.txt')))

    if herstr.lower() in message.content.lower():
        await channel.send("At least until operation: kill wife and kids")

    if kartstr.lower() in message.content.lower():
        await channel.send("**Official We Suck Mario Kart Ranking** - 8 > 64 > SNES > DD > 7 > Wii > DS > GBA")
    
    if mariostr.lower() in message.content.lower():
        await channel.send("**Official We Suck Super Mario Ranking** - Odyssey > 64 > World > 3 > World 2 = Galaxy = Galaxy 2 = Fury > 3D World > 3D Land > Sunshine > 1 > 2 = Land 2 > New DS = New Wii > New U > New 2 > Land > Lost")
    
    if ff2str.lower() in message.content.lower():
        await channel.send("The thing about civilization is that we are all 72 hours away from pure cannibalistic anarchy. That clock gets reset everytime we eat, everytiem we sleep but all of life as know it are on a precipice. FF2 was about 48 hrs for me. Everything you know and care about means nothing. That's the reality of culture and civilzation. It's an absolute cosmic shadow held up by essentially nothing. Final fantasy 2 taught me that.")
    
    if typongstr.lower() in message.content.lower():
       await channel.send("Don't make fun of my typong")
    
    if ffstr.lower() in message.content.lower():
        await channel.send("**Official We Suck Final Fantasy Ranking** - FF7R > FF6 > FF7 > FF9 > FF10 > FF15 > FF4 > FF12 > FF5 > FF10-2 > FF3 > FF1 > FF8 > MQ > FF2 > FF13")

    if neatostr.lower() in message.content.lower():
       await channel.send("neato burrito")

    if zeldastr.lower() in message.content.lower():
       await channel.send("**Official We Suck Zelda Ranking** - BotW > Minish > ALBW > ALttP > LA = WW > OoT > Oracle > MM > LoZ > TP > SS > AoL = PH > FSA > ST = TFH")
    
    if titwstr.lower() in message.content.lower():
        await channel.send("This is the way.")
    
    if daystr.lower() in message.content.lower():
        await channel.send("Tell me about it")
        
    # this keeps us from getting stuck in this function
    await client.process_commands(message)

client.run(token)
