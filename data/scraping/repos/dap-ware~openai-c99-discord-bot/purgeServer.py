#!/usr/bin/env python3
import asyncio
import discord
from discord.ext import commands
import yaml
import requests
import aiohttp
import openai
from c99api import EndpointClient
import logging
import sys
# Import additional classes from logging
from logging import FileHandler, Formatter
from logging.handlers import RotatingFileHandler

# Create a logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the logging level

# Create a rotating file handler
file_handler = RotatingFileHandler(
    "bot.log", maxBytes=2000000, backupCount=10
)  # Log messages will go to the "bot.log" file, max size is 2MB and keep up to 10 old log files
file_handler.setLevel(logging.INFO)  # Set the logging level for the file handler

# Create a formatter
formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add the formatter to the file handler
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

# Create a stream handler that logs to stdout
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

# Use the same formatter for the stream handler
stream_handler.setFormatter(formatter)

# Add the stream handler to the logger
logger.addHandler(stream_handler)

api = EndpointClient

# Load the secrets from the environment variables
try:
    with open("secrets.yaml", "r") as file:
        secrets = yaml.safe_load(file)
except Exception as e:
    logging.exception("An error occurred while reading secrets file. Exiting.")
    sys.exit(1)  # Stop the program

TOKEN = secrets["token"]
api.key = secrets["c99_api_key"]
openai.api_key = secrets["openai_api_key"]

# You might need to adjust the intents depending on what your bot needs
intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix="!", intents=intents)

# This set will store the IDs of channels where the purge command is currently running
purging_channels = set()


@client.event
async def on_ready():
    logging.info(f"We have logged in as {client.user}")


@client.command()
@commands.has_permissions(manage_messages=True)
async def purge(ctx):
    logging.info(f"Received purge command for channel: {ctx.channel.id}")

    if ctx.channel.id in purging_channels:
        logging.info(f"Purge command is already running in channel {ctx.channel.id}")
        return

    purging_channels.add(ctx.channel.id)
    logging.info(f"Started purging messages in channel {ctx.channel.id}")

    delay = 0.35  # initial delay is 350 ms
    message_count = 0

    try:
        async for message in ctx.channel.history(limit=None):
            while True:
                try:
                    await message.delete()
                    logging.info(
                        f"Deleted message {message.id} in channel {ctx.channel.id}"
                    )
                    await asyncio.sleep(delay)
                    message_count += 1

                    if message_count % 20 == 0:
                        if delay < 0.65:
                            delay = 0.65  # increase delay to 650 ms
                        elif delay < 1.2:
                            delay = 1.2  # increase delay to 1200 ms
                        logging.info(
                            f"Waiting for 20 seconds after deleting 20 messages in channel {ctx.channel.id}"
                        )
                        await asyncio.sleep(20)
                    break
                except discord.HTTPException:
                    logging.warning(
                        f"Rate limited while deleting message {message.id} in channel {ctx.channel.id}, waiting for 10 seconds"
                    )
                    await asyncio.sleep(10)
                except discord.NotFound:
                    logging.warning(
                        f"Message {message.id} not found in channel {ctx.channel.id}"
                    )
                    break
    except StopAsyncIteration:
        logging.info(
            f"No more messages to delete in channel {ctx.channel.id}, stopping all purge commands"
        )
        purging_channels.clear()
    finally:
        if ctx.channel.id in purging_channels:
            logging.info(f"Stopped purging messages in channel {ctx.channel.id}")
            purging_channels.remove(ctx.channel.id)


@client.command(description="Sends a gif based on the given keyword.")
async def gif(ctx, *, keyword):
    logging.info(f"Fetching gif for keyword: {keyword}")
    try:
        response = api.gif(keyword=keyword, json=True)
        if response["success"]:
            for url in response["images"]:
                embed = discord.Embed()
                embed.set_image(url=url)
                await ctx.send(embed=embed)
        else:
            error_message = f"An error occurred: {response['error']}"
            await ctx.send(error_message)
            logging.error(error_message)
    except Exception as e:
        logging.exception("An error occurred while fetching gif.")


@client.command(name="phone", description="Returns information about a phone number.")
async def phonelookup(ctx, *, number):
    logging.info(f"Fetching information for phone number: {number}")
    try:
        response = api.phonelookup(number, json=True)
        if response["success"]:
            await ctx.send(
                f"Country: {response['country']}\nCarrier: {response['carrier']}\nLine type: {response['line_type']}"
            )
        else:
            error_message = f"An error occurred: {response['error']}"
            await ctx.send(error_message)
            logging.error(error_message)
    except Exception as e:
        logging.exception("An error occurred while fetching phone number information.")


@client.command(name="email", description="Checks if an email address exists.")
async def emailvalidator(ctx, *, email):
    logging.info(f"Validating email address: {email}")
    try:
        response = api.emailvalidator(email, json=True)
        if response["success"]:
            if response["exists"]:
                await ctx.send("The email exists.")
            else:
                await ctx.send("The email does not exist.")
        else:
            error_message = f"An error occurred: {response['error']}"
            await ctx.send(error_message)
            logging.error(error_message)
    except Exception as e:
        logging.exception("An error occurred while validating email.")


@client.command(name="ports", description="Scans a host for open ports.")
async def portscanner(ctx, *, host):
    logging.info(f"Received portscanner command for host: {host}")
    try:
        response = api.portscanner(host, json=True)
        logging.info(f"Portscanner API response: {response}")

        if response["success"]:
            if "open_ports" in response:
                ports_message = (
                    f"Open ports: {', '.join(map(str, response['open_ports']))}"
                )
                await ctx.send(ports_message)
                logging.info(ports_message)
            else:
                no_ports_message = "No open ports found."
                await ctx.send(no_ports_message)
                logging.info(no_ports_message)
        else:
            error_message = f"An error occurred: {response['error']}"
            await ctx.send(error_message)
            logging.error(error_message)
    except Exception as e:
        logging.exception("An error occurred while processing the portscanner command.")


@client.command(name="subs", description="Gathers subdomains for a given domain")
async def subdomains(ctx, *, domain):
    logging.info(f"Received subdomains command for domain: {domain}")
    try:
        response = api.subdomainfinder(domain=domain, json=True)
        logging.info(f"Subdomain finder API response: {response}")

        if response["success"]:
            subdomains = [
                f"{subdomain['subdomain']}" for subdomain in response["subdomains"]
            ]
            subdomains_str = "\n".join(subdomains)

            logging.info(f"Writing subdomains to file: {domain}_subdomains.txt")
            with open(f"{domain}_subdomains.txt", "w") as file:
                file.write(subdomains_str)

            logging.info(f"Reading subdomains file: {domain}_subdomains.txt")
            with open(f"{domain}_subdomains.txt", "rb") as file:
                subdomains_message = f"Subdomains for {domain}:"
                await ctx.send(
                    subdomains_message,
                    file=discord.File(file, f"{domain}_subdomains.txt"),
                )
                logging.info(subdomains_message)
        else:
            error_message = f"Could not find subdomains for {domain}."
            await ctx.send(error_message)
            logging.error(error_message)
    except Exception as e:
        logging.exception("An error occurred while processing the subdomains command.")


@client.command()
async def ai(ctx, *, prompt):
    logging.info(f"Received AI command with prompt: {prompt}")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        logging.info(f"AI response: {response['choices'][0]['message']['content']}")

        response_message = f"{ctx.author.mention}, here's your response:\n>>> **{response['choices'][0]['message']['content']}**"
        await ctx.send(response_message)
        logging.info(response_message)
    except Exception as e:
        logging.exception("An error occurred while processing the AI command.")


@client.command(name="screenshot", description="Takes a screenshot of a given URL.")
async def screenshot(ctx, *, url):
    logging.info(f"Received screenshot command for url: {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.c99.nl/createscreenshot?key={secrets['c99_api_key']}&url={url}&json=true"
            ) as resp:
                response = await resp.json()
                logging.info(f"Screenshot API response: {response}")

        if response["success"]:
            embed = discord.Embed()
            embed.set_image(url=response["url"])
            await ctx.send(embed=embed)
            logging.info(f"Sent embed with screenshot url: {response['url']}")
        else:
            error_message = (
                f"An error occurred: {response.get('error', 'Unknown error')}"
            )
            await ctx.send(error_message)
            logging.error(error_message)
    except Exception as e:
        logging.exception("An error occurred while processing the screenshot command.")


@client.command(name="tor", description="Checks if an IP address is a Tor node.")
async def tor(ctx, *, ip):
    logging.info(f"Checking if IP {ip} is a Tor node.")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.c99.nl/torchecker?key={secrets['c99_api_key']}&ip={ip}&json=true"
            ) as resp:
                response = await resp.json()

        if response["success"]:
            if response["result"]:
                message = f"The IP address {ip} is a Tor node."
                await ctx.send(message)
                logging.info(message)
            else:
                message = f"The IP address {ip} is not a Tor node."
                await ctx.send(message)
                logging.info(message)
        else:
            error_message = (
                f"An error occurred: {response.get('error', 'Unknown error')}"
            )
            await ctx.send(error_message)
            logging.error(error_message)
    except Exception as e:
        logging.exception("An error occurred while checking if IP is a Tor node.")


"""
RUN THE CLIENT
"""
client.run(TOKEN)
