#!/usr/bin/env python3

from dotenv import load_dotenv
import asyncio
import json
import discord
from discord import app_commands
import os
from termcolor import colored
import datetime
import pandas as pd
from pandasai import SmartDataframe, SmartDatalake
from langchain.chat_models import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor
import logging
logging.basicConfig()
logging.getLogger("pandasai").setLevel(logging.DEBUG)

pd.set_option("display.max_columns", None)
load_dotenv()

# No special intents required for slash commands
client = discord.Client(intents=None)
# This handles slash commands
tree = discord.app_commands.CommandTree(client)

BOT_MEMORY_FILE = "bot_memory.json"

# Load memory, or start with a fresh memory dictionary
try:
    with open(BOT_MEMORY_FILE) as f:
        memory = json.load(f)
except FileNotFoundError:
    memory = {}
except json.decoder.JSONDecodeError:
    print(colored(f"Error: {BOT_MEMORY_FILE} is not valid JSON", "red"))
    memory = {}


def save_memory():
    print("Saving memory: ", memory)
    with open(BOT_MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4, sort_keys=True)


# Works for both guild channels and DMs
async def get_channel(channel):
    if channel["type"] == "DM":
        user = await client.fetch_user(channel["id"])
        channel = await user.create_dm()
    else:
        channel = client.get_channel(channel["id"])
    return channel


@client.event
async def on_ready():
    await tree.sync()  # Sync available commands
    # DM neonninja and let him know the bot is online
    user = await client.fetch_user(200793152167084034)
    channel = await user.create_dm()
    await channel.send("I'm back online!")
    # This is a background task that runs every 15 minutes
    while True:
        print("Checking for new exchange rate...")
        df = pd.read_csv("exchange_rates.csv")
        rate = df.iloc[-1].CubeToQred
        if rate != memory.get("last_exchange_rate"):
            for channel in memory.get("exchange_rate_channels", []):
                channel = await get_channel(channel)
                alert_role = None
                if rate > 7000:
                    alert_role = discord.utils.get(channel.guild.roles, name="alerts")
                await channel.send(
                    f"Today's exchange rate is {rate} Qredits per Qube {alert_role.mention if alert_role else ''}"
                )
            memory["last_exchange_rate"] = int(rate)
            save_memory()
        print("Checking for new MAZ...")
        df = pd.read_csv("battlestats.csv")
        latest_date = df.Date.max()
        if latest_date != memory.get("last_MAZ_date"):
            df = df[df.Date == latest_date]
            for channel in memory.get("MAZ_channels", []):
                channel = await get_channel(channel)
                await channel.send(format_MAZ(df))
            memory["last_MAZ_date"] = latest_date
            save_memory()
        print("Checking for new nano range...")
        range_km = get_range()
        if range_km != memory.get("last_range"):
            for channel in memory.get("nano_range_channels", []):
                channel = await get_channel(channel)
                if range_km > 7000:
                    alert_role = discord.utils.get(channel.guild.roles, name="alerts")
                await channel.send(
                    f"Today's nanomissile range is {range_km} km {alert_role.mention if alert_role else ''}"
                )
            memory["last_range"] = range_km
            save_memory()
        await asyncio.sleep(60 * 15)


registerable_things = [
    app_commands.Choice(name="Exchange rates", value="exchange_rate"),
    app_commands.Choice(name="Most Active Zones", value="MAZ"),
    app_commands.Choice(name="Nanomissile range changes", value="nano_range"),
]


@tree.command(
    name="register",
    description="Register this channel for daily exchange rate updates, MAZ updates or nanomissile range updates",
)
@app_commands.choices(option=registerable_things)
async def register(interaction: discord.Interaction, option: app_commands.Choice[str]):
    if type(interaction.channel) is discord.channel.DMChannel:
        channel = {
            "type": "DM",
            "id": interaction.user.id,
            "name": interaction.user.name,
        }
    elif type(interaction.channel) is discord.channel.TextChannel:
        channel = {
            "type": "Text",
            "id": interaction.channel_id,
            "name": interaction.channel.name,
            "guild": interaction.guild.name,
        }
    key = f"{option.value}_channels"
    memory[key] = memory.get(key, []) + [channel]
    save_memory()
    await interaction.response.send_message("✅")


@tree.command(
    name="unregister",
    description="Un-register this channel for daily exchange rate updates, MAZ updates or nanomissile range updates",
)
@app_commands.choices(option=registerable_things)
async def unregister(
    interaction: discord.Interaction, option: app_commands.Choice[str]
):
    if type(interaction.channel) is discord.channel.DMChannel:
        channel_id = interaction.user.id
    elif type(interaction.channel) is discord.channel.TextChannel:
        channel_id = interaction.channel_id
    key = f"{option.value}_channels"
    memory[key] = [c for c in memory.get(key, []) if c["id"] != channel_id]
    save_memory()
    await interaction.response.send_message("✅")


@tree.command(name="ping", description="Respond with pong")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message("pong")


@tree.command(
    name="exchange_rate",
    description="Respond with the current Qube to Qredits exchange rate",
)
async def exchange_rate(interaction: discord.Interaction):
    df = pd.read_csv("exchange_rates.csv")
    rate = df.iloc[-1].CubeToQred
    await interaction.response.send_message(
        f"The current exchange rate is {rate} Qredits per Qube"
    )


colormap = {
    "Swarm": "green",
    "Legion": "red",
    "Faceless": "magenta",
}


def format_MAZ(df=None):
    if df is None:
        df = pd.read_csv("battlestats.csv")
    latest_date = df.Date.max()
    df = df[df.Date == latest_date]
    result = f"Most Active Zones for {latest_date} QST:\n\n"
    for i, row in df.iterrows():
        link = f'https://portal.qonqr.com/Home/BattleStatistics/{row["Battle Report Number"]}'
        result += f'[{row["Zone Name"]}, {row["Region"]}, {row["Country"]}](<{link}>)\n'
    assert len(result) <= 2000
    return result


@tree.command(name="maz", description="Show today's Most Active Zones")
async def maz(interaction: discord.Interaction):
    await interaction.response.send_message(format_MAZ())


format_MAZ()


def get_range():
    range_km = 1609.34
    if datetime.datetime.utcnow().weekday() == 0:  # 0 corresponds to Monday
        week = datetime.datetime.utcnow().day // 7
        if week == 1:
            range_km = 4828.032
        elif week == 2:
            range_km = 9656.064
        elif week >= 3:
            range_km = 20116.8
    return range_km


@tree.command(
    name="nanomissile_range", description="Check the current nanomissile range in km"
)
async def nano_range(interaction: discord.Interaction):
    await interaction.response.send_message(
        f"Today's nanomissile range is {get_range()} km"
    )

def ai_query(prompt):
    df = SmartDataframe("battlestats.csv", config={
        "name": "battles",
        "description": "MAZ battles",
        "llm": ChatOpenAI(),
        "enable_cache": False,
        "custom_whitelisted_dependencies": ["PIL"]
    }).drop(columns=["players"])
    player_df = SmartDataframe("battlestats_players.csv", config={
        "name": "players",
        "description": "players who fought in the MAZ battles",
        "llm": ChatOpenAI(),
        "enable_cache": False,
        "custom_whitelisted_dependencies": ["PIL"]
    })
    player_details_df = SmartDataframe("player_details.csv", config={
        "name": "player details",
        "description": "Details about players",
        "llm": ChatOpenAI(),
        "enable_cache": False,
        "custom_whitelisted_dependencies": ["PIL"],
    })
    df = SmartDatalake([df, player_df, player_details_df], config={"llm": ChatOpenAI(), "enable_cache": False, "max_retries": 10})
    result = df.chat(prompt)
    return result, df.last_result

@tree.command(
    name="ai", description="Natural language MAZ query using AI magic ✨"
)
async def ai(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()
    loop = asyncio.get_event_loop()
    try:
        result, last_result = await loop.run_in_executor(ThreadPoolExecutor(), ai_query, prompt)
        print(result, last_result)
        if last_result:
            if last_result["type"] == "plot":
                await interaction.followup.send(file=discord.File(last_result["value"]), content=f"Prompt: {prompt}. Result:")
            elif last_result["type"] == "dataframe":
                await interaction.followup.send(f"Prompt: {prompt}. Result:\n```{last_result['value'].to_markdown()}```")
            else:
                await interaction.followup.send(f"Prompt: {prompt}. Result:\n{last_result['value']}")
        else:
            await interaction.followup.send(f"Prompt: {prompt}. Result:\n{result}")
    except:
        await interaction.followup.send(f"Prompt: {prompt}. Result:\n💀")

client.run(os.getenv("DISCORD_BOT_TOKEN"))
