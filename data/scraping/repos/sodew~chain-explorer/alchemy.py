import discord
import os
import json
import requests
import aiohttp
import re
import datetime
import openai
from pprint import pprint
from web3 import Web3
import pickle

from openai.error import RateLimitError
import asyncio

# Load token directly from tokens.json
with open('tokens.json') as f:
    tokens = json.load(f)
    discord_token = tokens['discord']
    # Assuming you've added your OpenAI API key to your tokens.json
    openai_token = tokens['openai']
    passport_token = tokens['passport']
    scorer_token = tokens['scorer']

# Setup Alechmy API
alchemy_url = "https://eth-mainnet.g.alchemy.com/v2/kyvyHty9Uu7gLaas166z5rFBPsxQDDqT"
w3 = Web3(Web3.HTTPProvider(alchemy_url))

# Setup Passport API
if passport_token:
    passport_headers = {
        'Content-Type': 'application/json',
        'X-API-Key': passport_token
    }

# Setup OpenAI API
openai.api_key = openai_token


def parse_api_response(response):
    wallet_address = response["story"]["walletId"]
    ens_domain = response["story"]["ensName"]
    creation_date = datetime.datetime.fromtimestamp(
        response['story']['walletDOBTimestamp']).strftime("%B %d, %Y")
    latest_transaction_date = datetime.datetime.fromtimestamp(
        response['story']['latestTransactionDateTimestamp']).strftime("%B %d, %Y")

    passport = ""
    print(response["passport"])
    if response["passport"]:
        passport_score = response["passport"]["score"]
        passport_timestamp = datetime.datetime.fromisoformat(
            response["passport"]["last_score_timestamp"]).strftime("%B %d, %Y")
        passport = f"They have a Gitcoin Passport score of {passport_score} as of {passport_timestamp}"
    else:
        print("no passport response")

    number_of_nfts = response["story"]["numberOfNftsOwned"]

    def generate_title_list(data):
        if data:
            return '\n'.join([f"• {ach['title']}" for ach in data])
        else:
            return "No data found."

    def generate_description_list(data):
        if data:
            return '\n'.join([f"• {ach['description']}" for ach in data])
        else:
            return "No data found."

    def generate_info_list(data):
        if data:
            return '\n'.join([f"• {ach['title']} : {ach['description']}" for ach in data])
        else:
            return "No achievements found."

    nft_achievements = generate_description_list(
        response["story"]["nftAchievements"])
    defi_achievements = generate_description_list(
        response["story"]["deFiAchievements"])
    community_achievements = generate_description_list(
        response["story"]["communityAchievements"])
    vibe_achievements = generate_info_list(
        response["story"]["vibeAchievements"])

    return (
        f"The wallet {wallet_address} belongs to {ens_domain}. This wallet was "
        f"created on {creation_date} and their latest transaction was on {latest_transaction_date}.\n\n"
        f"They own {number_of_nfts} NFTs including:\n{nft_achievements}\n\n"
        f"Evidence of their participation in DeFi and money markets:\n{defi_achievements}\n\n"
        f"Evidence of participation in web3 communities:\n{community_achievements}\n\n"
        f"Evidence of engagements within the web3 ecosystem:\n{vibe_achievements}\n\n"
        f"{passport}"
    )


def parse_passport(response):
    items = []
    for item in response["items"]:
        if item['metadata']:
            name = item['metadata']['name']
            desc = item['metadata']['description']
            pprint(name)
            items.append(f"* **{name}**: {desc}")

    return (
        f"They own:\n"
        + "\n".join(items))


class ChatBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_responses = {}  # A dictionary to cache API responses

    async def on_ready(self):
        print("Bot is ready!")

    async def on_message(self, message):
        if message.author == self.user:  # Ignore bot's own messages
            return

        if message.content.startswith("!alchemy"):
            # print("is connected? " + w3.is_connected())
            await message.channel.send(f"is connected? " + str(w3.is_connected()))
            await message.channel.send(f"ETH balance: " + str(w3.eth.get_balance('0x475Eaa9b5386F2fD85D821CF72eec45FE7E4c09a')))

        if message.content.startswith("!cache"):
            print("CACHE: \n")
            pprint(self.api_responses)
            if not self.api_responses:
                return None

            addresses = list(self.api_responses.keys())
            # TODO change address to the last key in cache
            await message.channel.send(f"addresses: {addresses}")
            await message.channel.send(f'latest wallet address = {addresses[-1]}')

            # await message.channel.send(self.api_responses[])

        if message.content.startswith('tell me about '):
            address = message.content.split(" ")[3]
            await message.channel.send(f"Fetching on-chain data from {address}. This may take a moment...")

            # Check if the data for this address is already in cache
            if address in self.api_responses:
                await message.channel.send(parse_api_response(self.api_responses[address]))
                return

            CHAINSTORY_URI = f"https://www.chainstory.xyz/api/story/getStoryFromCache?walletId={address}"

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(CHAINSTORY_URI) as response:
                        if response.status != 200:
                            await message.channel.send(f"Error {response.status}: Unable to retrieve information for the provided ethereum address.")
                            return

                        data = await response.json()

                if data.get('success') and data.get('story'):
                    pprint(data)
                    self.api_responses[address] = data

                    # Retrieve more data from Gitcoin Passport
                    passport_address = data["story"]["walletId"]
                    GET_PASSPORT_SCORE_URI = f"https://api.scorer.gitcoin.co/registry/v2/score/698/{passport_address}"

                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(GET_PASSPORT_SCORE_URI, headers=passport_headers) as passport_response:
                                if passport_response.status == 200:
                                    passport_data = await passport_response.json()
                                    f"User {address} has a passport score"

                                    # Adding the passport score data into the nested dictionary
                                    self.api_responses[address]['passport'] = passport_data
                                else:
                                    self.api_responses[address]['passport'] = {
                                    }
                                    print(
                                        f"Error {passport_response.status}: Unable to retrieve passport score for the address.")

                    except Exception as e:
                        await message.channel.send(f"Error fetching passport score: {str(e)}")

                    # with open('local_state.pkl', 'wb') as f:
                    #     pickle.dump(self.api_responses, f)
                    await message.channel.send(parse_api_response(self.api_responses[address]))
                else:
                    await message.channel.send("Unable to retrieve chain history for the provided ENS domain.")

            except Exception as e:
                error_msg = f"Error fetching data: {str(e)}"
                print(error_msg)
                await message.channel.send(error_msg)

        if "passport stamps" in message.content.lower():
            cached_addresses = list(self.api_responses.keys())
            latest_address = {cached_addresses[-1]}

            # await message.channel.send(f"Fetching data from {latest_address}")
            # Check if the data for this ENS domain is already in cache
            if address in self.api_responses:
                await message.channel.send(f"{latest_address} data in database")
                # TODO: FIX
                passport_data_chunks = parse_passport(data)
                for chunk in passport_data_chunks:
                    await message.channel.send(chunk)
                # await message.channel.send(parse_passport(self.api_responses[address]))
                return

            GET_PASSPORT_STAMPS_URI = f"https://api.scorer.gitcoin.co/registry/v2/stamps/{latest_address}?limit=1000&include_metadata=true"

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(GET_PASSPORT_STAMPS_URI, headers=passport_headers) as response:
                        if response.status != 200:
                            await message.channel.send(f"Error {response.status}: Unable to retrieve passport for the provided address.")
                            return

                        data = await response.json()
                        pprint(data)
                        # Check if the data is not None
                        if data is not None:
                            await message.channel.send(f"Successfully got passport data!")
                            # await message.channel.send(parse_passport(data))
                            passport_data_chunks = parse_passport(data)
                            for chunk in passport_data_chunks:
                                await message.channel.send(chunk)
                        else:
                            await message.channel.send(f"Error: Passport data is None")

            except Exception as e:
                await message.channel.send(f"Error fetching data: {str(e)}")

        if message.content.startswith("what is "):
            query = message.content[len("what is "):]
            prompt = f"I want you to act as a blockchain expert. Explain {query} "
            try:
                response = openai.Completion.create(
                    model="text-davinci-003", prompt=prompt, temperature=0.6, max_tokens=200)
                await message.channel.send(response.choices[0].text.strip())
            except RateLimitError:
                await message.channel.send("Sorry, I'm getting too many requests right now. Please try again later.")
                # Introducing a delay. Adjust as needed.
                await asyncio.sleep(10)


if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True
    bot = ChatBot(intents=intents)
    bot.run(discord_token)
