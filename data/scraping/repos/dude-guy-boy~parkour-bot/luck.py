# luck.py

from interactions import (
    Client,
    Extension,
    listen,
    events,
    Task,
    IntervalTrigger,
    Message
    )
import src.logs as logs
from random import choice
from random import randint
from src.database import UserData
from lookups.luckreference import powers
from lookups.aipersonalities import personalities
from src.openai import OpenAI, Role, Message as AIMessage

class Luck(Extension):

    def __init__(self, client: Client):
        self.client = client
        self.logger = logs.init_logger()
        self.save_luck_cache.start()
        self.luck_cache = {}

    @Task.create(IntervalTrigger(minutes=5))
    async def save_luck_cache(self):
        self.logger.info("Saving Luck Cache")

        # Update every users luck data
        for user in self.luck_cache:
            data = UserData.get_user(id=user)

            if not data:
                data = {"message_count" : 0, "lucky_points" : 0, "messages": []}

            data = {
                'message_count': data['message_count'] + self.luck_cache[user]['message_count'],
                'lucky_points': data['lucky_points'] + self.luck_cache[user]['lucky_points'],
                'messages': data['messages'] + self.luck_cache[user]['messages']
            }

            UserData.set_user(id=user, data=data)

        # Clear cache
        self.luck_cache = {}

    @listen("on_message_create")
    async def on_message(self, event: events.MessageCreate):
        message = event.message

        # Ignore if sent by bot
        if message.author.bot:
            return

        # Calculate message luck
        random_number = randint(0, 1000000000000)

        luck_power = 0

        for power in range(3, 10):
            if random_number % (10 ** power) == 0:
                luck_power = power

        # Initialise luck profile in cache if it doesnt exist
        luck_profile = None
        if str(message.author.id) not in self.luck_cache:
            luck_profile = {"message_count" : 0, "lucky_points" : 0, "messages": []}
        else:
            luck_profile = self.luck_cache[str(message.author.id)]

        luck_profile["message_count"] += 1

        # Check luck of the message
        if luck_power in powers:
            luck_profile = await self.process_luck(message=message, luck_profile=luck_profile, luck_power=luck_power)

        # Set in cache
        self.luck_cache[str(message.author.id)] = luck_profile

    async def process_luck(self, message: Message, luck_profile, luck_power: int):
        reference = powers[luck_power]
        
        for emoji in reference['reactions']:
            try:
                await message.add_reaction(emoji)
            except:
                pass

        for idx, response in enumerate(reference['replies']):
            if response == "ai":
                response = OpenAI.chat_response(messages=[AIMessage(Role.SYSTEM, f"You are a {choice(personalities)} responding in shock to someone who's message got one in {reference['text']} luck. Don't mention the lottery")]
                                                , max_tokens=150)
            
            if idx == 0:
                await message.reply(response)
            else:
                await message.channel.send(response)

        luck_profile["lucky_points"] += reference['points']
        luck_profile["messages"].append(
            {"odds": reference['text'], "message_id": int(message.id), "channel_id": int(message.channel.id), "content": str(message.content)})
        
        self.logger.info(f"{message.author.username} just got one in {reference['text']} luck")
        await logs.DiscordLogger.log_raw(bot=self.bot, description=f"{message.author.username} just got [one in {reference['text']}]({message.jump_url}) luck")

        return luck_profile

    # TODO: Add command that scans all channels to see which have perms for everyone to view.
    # Channels not on this list should have the content listed as blocked

    # TODO: Luck profiles

def setup(bot):
    # This is called by interactions.py so it knows how to load the Extension
    Luck(bot)