import discord
import openai
from discord.ext import commands, tasks
import random
from utils.chat_openai import chat
from utils.log_config import setup_logging, logging
import time
from Commands.Admin.persona import get_persona, get_persona_by_name


setup_logging
logger = logging.getLogger(__name__)

BREEZE_HEART = "<a:z8_leafheart_excited:806057904431693824>"
ABBY_RUN = "<a:Abby_run:1135375927589748899>"
ABBY_IDLE = "<a:Abby_idle:1135376647495884820>"
ABBY_JUMP = "<a:Abby_jump:1135372059350933534>"

class RandomMessages(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None: 
        self.bot = bot
        self.send_random_message.start()
        self.channel_id = 802512963519905852
        self.abby_chat = 1103490012500201632
        self.last_message_timestamp = None

    def cog_unload(self):
        # This will cancel the task when the cog is unloaded
        self.send_random_message.cancel()

    @commands.Cog.listener()
    async def on_ready(self):
        logger.info("Random Messages cog is ready")


    # Task to send random message every 3 hours (if the previous messages aren't sent by the bot)
    @tasks.loop(hours=12)
    async def send_random_message(self):
        await self.bot.wait_until_ready()
        logger.info("Checking if we should send a random message")
        channel = self.bot.get_channel(self.channel_id)
        
        last_message = None
        async for message in channel.history(limit=1):
            last_message = message
            break  # Since we only need the last message, we break after the first iteration

        if last_message and last_message.author.id != self.bot.user.id:
            logger.info("Sending random message")
            msg = await channel.send(await self.openai_chat()) 
            await msg.add_reaction(BREEZE_HEART)
            self.last_message_timestamp = time.time()
        else:
            logger.info("Last message was sent by the bot, skipping")

    @commands.command(name='randommessage', help='Sends a random message to the Breeze Club')
    async def randommessage_command(self, ctx):
        """Command to manually trigger a random message."""
        channel = self.bot.get_channel(self.channel_id)
        # Check if the last message was not sent by the bot
        last_message = None
        async for message in channel.history(limit=1):
            last_message = message
            break
        
        if last_message and last_message.author.id != self.bot.user.id:
            logger.info("Sending random message due to command")
            msg = await channel.send(await self.openai_chat())
            await msg.add_reaction(BREEZE_HEART)
        else:
            await ctx.send("Last message was sent by the bot. Skipping.")    

    def get_random_message(self):
        """
        Selects and returns a random message based on weighted random selection.
        
        Each message in the 'messages' list has an associated weight. Messages with higher weights
        are more likely to be selected. The function computes the total weight, selects a random number
        between 0 and the total weight, and then iterates through the messages to find the one corresponding 
        to the random number based on cumulative weights.
        
        Args:
            self: The instance of the object calling this method.
        
        Returns:
            str: A randomly selected message based on its weight.
        """
        
        messages = [
            ("Ever explored a new skill you wanted to master?", 1),
            ("Tell us about a documentary that changed the way you think.", 1),
            ("Are there any podcasts you're currently hooked on?", 1),
            ("Which board game is your all-time favorite?", 1),
            ("Ever been inspired by a biography or autobiography?", 1),
            ("What's your favorite way to spend a weekend evening?", 1),
            ("Content creators, plug your latest! And don't forget to mention your streaming times!", 2),
            ("Share a memorable moment you had in a game recently.", 1),
            ("Let's make our community stronger! Consider giving the Server a Boost.", 3),
            ("Stay connected with us: https://discord.com/servers/breeze-club-547471286801268777", 2),
            ("Welcome newcomers! What song always gets you on the dance floor?", 3),
            ("Think Breeze Club is cool? Share the vibe! https://discord.gg/yGsBGQAC49", 3),
            ("Time for some Fun Facts!", 4),
        ]


        total_weight = sum(weight for message, weight in messages)
        random_num = random.uniform(0, total_weight)
        cumulative_weight = 0
        for message, weight in messages:
            cumulative_weight += weight
            if random_num < cumulative_weight:
                return message

    def get_random_emoji(self):
        emojis = [
            ABBY_IDLE,
            ABBY_RUN,
            ABBY_JUMP
        ]
        return random.choice(emojis)

    
    async def openai_chat(self):
        active_persona_doc = get_persona()
        active_persona = active_persona_doc['active_persona'] if active_persona_doc else 'bunny'
        persona_message = get_persona_by_name(active_persona)['persona_message']

        model = "gpt-3.5-turbo"
        retry_count = 0
        while retry_count < 3:
            try:
                random_message = self.get_random_message()
                logger.info(f"Selected random topic: {random_message}")  # Log the selected topic            
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": persona_message},
                        {"role": "system", "content":  f"Create a short discord message for the Breeze Club"},
                        {"role": "system", "content": f"Topic: {random_message}"}
                    ],
                    max_tokens=225,
                    temperature=0.6
                )

                status_code = response["choices"][0]["finish_reason"]
                assert status_code == "stop", f"The status code was {status_code}."
                return response["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(
                    f"An error occurred while processing the summarize request: {str(e)}")
                logger.info("Retrying...")
                time.sleep(1)
                retry_count += 1

        return "Oops, something went wrong. Please try again later."

# async def setup(bot):
#     await bot.add_cog(RandomMessages(bot))
