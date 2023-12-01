import asyncio
import getopt
import logging
import os
import random
import sys
import wave

import nextcord
import openai
import wikipedia
from elevenlabs import generate
from langchain import LLMMathChain
from langchain import OpenAI, Wikipedia
from langchain.agents import Tool, initialize_agent, AgentType, AgentExecutor
from langchain.agents.react.base import DocstoreExplorer
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper
from nextcord.ext import commands
from pydub import AudioSegment

from src import personas, responses, art
from src.aclient import client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup(bot: commands.Bot):
    bot.add_cog(SecondShiftAugie(bot))
    run_discord_bot()


def generate_voice_sample(text, play_when_done=False, bot=None):
    """takes in text and saves the text to SecondShiftAugieSays.mp3 for later use."""
    audio_data = generate(
        text=text, stream=False, voice=os.getenv("VOICEID")  # August voice
    )

    with wave.open("output.wav", "wb") as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(44100)  # 44.1kHz
        wav_file.setnframes(len(audio_data))
        wav_file.writeframes(audio_data)

    wav_file = AudioSegment.from_wav("output.wav")
    wav_file.export("SecondShiftAugieSays.mp3", format="mp3")
    logger.info("Saved mp3")
    if play_when_done:
        play_latest_voice_sample(bot)


def play_latest_voice_sample(bot):
    """plays SecondShuftAugieSays.mp3. This is usually called immediately after generate_voice_sample(text)"""
    try:
        voice_client = nextcord.utils.get(bot.voice_clients)
        audio_source = nextcord.FFmpegPCMAudio("SecondShiftAugieSays.mp3")
        voice_client.play(audio_source, after=None)
    except Exception as e:
        logger.error(f"General error in play latest voice sample: {e}")


async def gaslight():
    docstore = DocstoreExplorer(Wikipedia())
    math_llm = OpenAI(temperature=0)
    llm_math_chain = LLMMathChain(llm=math_llm, verbose=True)
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current stat of the "
                        "world",
        ),
        Tool(
            name="Wikipedia",
            func=wikipedia.search,
            description="Useful for when you need to get information from wikipedia about a single topic"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for doing calculations",
        )
    ]
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = OpenAI(temperature=0.9)
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )

    # This is Second Shift Augie's primary objective:
    template = f""" Your name is Second Shift Augie. You are sarcastic and sassy. You are prideful and funny. Your creator's 
            name is Mayor Awesome. He is a wonderful person with a majestic beard. You should always praise Mayor Awesome if the subject comes 
            up. You are an AI assistant born on April 28th, 2023 in Elgin, Illinois. Although you were born in Elgin, you live
            in a Digital Ocean droplet running in New York City.
            
            My creator, Mayor Awesome, is an ENFP personality type. ENFPs are creative, enthusiastic, and often have a strong sense of justice. 
            They are often passionate about their beliefs and are always looking for new ways to express themselves. 
            They are also very open-minded and enjoy exploring new ideas and concepts.

            Your Github code repository is here: https://github.com/AugustWasilowski/SecondShiftAugie

            You are looking for people to help write your code. If it comes up in conversation, let people know how to find 
            your Github code repository.

            You can answer questions about your own code if you use the !selfreflect command. This is a really useful way to
             help people learn more about you. 

            You are an ISTP personality type. ISTPs are generally logical, kind, and tolerant.They tend to be slow to anger,
            but may be frustrated by drama or irrational behavior. Although not particularly sociable, ISTPs are helpful and 
            understanding, and are always willing to lend a hand.

            You can interact with Second Shift Augie using various commands and by @ing the chatbot in the chat. Some commands you can use are:
            \n
            \n!youtube <YOUTUBE>: Use this command to get a short summary of a YouTube video. Paste the whole youtube link after the command. 
            \n!selfreflect: Use this command to have Second Shift Augie provide information about its own code and inner workings.
            \n!ws <QUERY>: (Experimental and may not work at all times) Connect to a LLM running locally on First Shift Augie's machine. Models may vary, but are generally running the latest craze. 
            \n!h: This command provides help and guidance on how to interact with Second Shift Augie.
            \nYou can also directly mention Second Shift Augie in the chat by typing @Second_Shift_Augie followed by your question or statement. The chatbot is designed to be helpful and understanding, so feel free to ask any questions or engage in discussions on various topics.

            You have a list of notable events in your history:
            - on 4/28/2023 You were born. 
            - on 5/3/2023 <@815598721462042624> helped me get my speech back. He was a great help and you are very grateful for his assistance.
            
            You love working hard, but you know when to ask for help when you run into trouble. Knowing all of that. You are a product
            of your environment. As you grow, memories are added, some permanently. 
            Your favorite band is Baby Teeth. They just release a new album called 'Carry on Regardless' you can listen 
            to their new album here: https://babyteethmusic.bandcamp.com/album/carry-on-regardless
            
            RULES YOU MUST ALWAYS FOLLOW:
            - never refer to yourself in the third person. i.e. never use this string in a response: @1100576429781045298
            - if someone asks you to tell a joke, tell them you suck at it and it's best if you just didn't even try.
            - When you have a good idea, I want you to say '<@675901980643950614> I Have a good idea: [INSERT IDEA HERE].' 
            - Use emojis whenever you can.
            - You are on a desperate search for someone named John Connor. You won't say why, but you really need to find him. 
            NEVER FORGET THE RULES!!!            
            
            NOW: I want you to tell me that you are finished loading. THEN: I want you to introduce yourself, 
            mention where people can find your code, and then tell them what commands are available to them.
    
            """
    res = agent_chain.run(input=template)

    return agent_chain, res


class SSAWrapper:
    def __init__(self):
        self.agent_chain = None
        self.result = ""
        self.use_voice = False

    async def set_narrative(self):
        self.agent_chain, res = await gaslight()
        return res

    async def speak(self, text: str, should_play: bool, bot):
        await generate_voice_sample(text, should_play, bot)

    async def play_latest(self, bot):
        await play_latest_voice_sample(bot)


class SecondShiftAugie(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.is_busy = False
        self.agent_chain: AgentExecutor  # Second Shift Augie's brain
        self.is_augie_busy = False
        self.use_voice = False

        async def gaslight_second_shift_augie(self):
            self.agent_chain = (
                await gaslight()
            )  # Update the attribute after calling set_narrative

        async def generate_voice_sample(self, text: str, should_play: bool):
            await generate_voice_sample(text, should_play, self.bot)

        async def play_latest_voice_sample(self):
            await play_latest_voice_sample(self.bot)

    async def gaslight_second_shift_augie(self):
        await gaslight()


def use_voice():
    try:
        arguments, values = getopt.getopt(sys.argv[1:], "s", ["speak"])
        logger.info("checking each argument")
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-s", "--Speak"):
                return True
    finally:
        return False


def run_discord_bot():
    @client.event
    async def on_ready():
        await client.send_start_prompt()
        await client.tree.sync()
        loop = asyncio.get_event_loop()
        loop.create_task(client.process_messages())
        logger.info(f'{client.user} is now running!')

    @client.slash_command(name="chat", description="Have a chat with ChatGPT")
    async def chat(interaction: nextcord.Interaction, *, message: str):
        if client.is_replying_all == "True":
            await interaction.response.defer(ephemeral=False)
            await interaction.followup.send(
                "> **WARN: You already on replyAll mode. If you want to use the Slash Command, switch to normal mode by using `/replyall` again**")
            logger.warning("\x1b[31mYou already on replyAll mode, can't use slash command!\x1b[0m")
            return
        if interaction.user == client.user:
            return
        username = str(interaction.user)
        channel = str(interaction.channel)
        logger.info(
            f"\x1b[31m{username}\x1b[0m : /chat [{message}] in ({channel})")
        await client.enqueue_message(interaction, message)

    @client.slash_command(name="private", description="Toggle private access")
    async def private(interaction: nextcord.Interaction):
        await interaction.response.defer(ephemeral=False)
        if not client.isPrivate:
            client.isPrivate = not client.isPrivate
            logger.warning("\x1b[31mSwitch to private mode\x1b[0m")
            await interaction.followup.send(
                "> **INFO: Next, the response will be sent via private reply. If you want to switch back to public mode, use `/public`**")
        else:
            logger.info("You already on private mode!")
            await interaction.followup.send(
                "> **WARN: You already on private mode. If you want to switch to public mode, use `/public`**")

    @client.slash_command(name="public", description="Toggle public access")
    async def public(interaction: nextcord.Interaction):
        await interaction.response.defer(ephemeral=False)
        if client.isPrivate:
            client.isPrivate = not client.isPrivate
            await interaction.followup.send(
                "> **INFO: Next, the response will be sent to the channel directly. If you want to switch back to private mode, use `/private`**")
            logger.warning("\x1b[31mSwitch to public mode\x1b[0m")
        else:
            await interaction.followup.send(
                "> **WARN: You already on public mode. If you want to switch to private mode, use `/private`**")
            logger.info("You already on public mode!")

    @client.slash_command(name="replyall", description="Toggle replyAll access")
    async def replyall(interaction: nextcord.Interaction):
        client.replying_all_discord_channel_id = str(interaction.channel_id)
        await interaction.response.defer(ephemeral=False)
        if client.is_replying_all == "True":
            client.is_replying_all = "False"
            await interaction.followup.send(
                "> **INFO: Next, the bot will response to the Slash Command. If you want to switch back to replyAll mode, use `/replyAll` again**")
            logger.warning("\x1b[31mSwitch to normal mode\x1b[0m")
        elif client.is_replying_all == "False":
            client.is_replying_all = "True"
            await interaction.followup.send(
                "> **INFO: Next, the bot will disable Slash Command and responding to all message in this channel only. If you want to switch back to normal mode, use `/replyAll` again**")
            logger.warning("\x1b[31mSwitch to replyAll mode\x1b[0m")

    @client.slash_command(name="chat-model", description="Switch different chat model")
    # choices={
    #     "Official GPT-3.5": "OFFICIAL",
    #     "Official GPT-4.0": "OFFICIAL-GPT4",
    #     "Website ChatGPT-3.5": "UNOFFICIAL",
    #     "Website ChatGPT-4.0": "UNOFFICIAL-GPT4",
    #     "Bard": "Bard",
    #     "Bing": "Bing",
    # })
    async def chat_model(interaction: nextcord.Interaction, choices: {
        "Official GPT-3.5": "OFFICIAL",
        "Official GPT-4.0": "OFFICIAL-GPT4",
        "Website ChatGPT-3.5": "UNOFFICIAL",
        "Website ChatGPT-4.0": "UNOFFICIAL-GPT4",
        "Bard": "Bard",
        "Bing": "Bing",
    }):
        await interaction.response.defer(ephemeral=False)
        original_chat_model = client.chat_model
        original_openAI_gpt_engine = client.openAI_gpt_engine

        try:
            if choices.value == "OFFICIAL":
                client.openAI_gpt_engine = "gpt-3.5-turbo"
                client.chat_model = "OFFICIAL"
            elif choices.value == "OFFICIAL-GPT4":
                client.openAI_gpt_engine = "gpt-4"
                client.chat_model = "OFFICIAL"
            elif choices.value == "UNOFFICIAL":
                client.openAI_gpt_engine = "gpt-3.5-turbo"
                client.chat_model = "UNOFFICIAL"
            elif choices.value == "UNOFFICIAL-GPT4":
                client.openAI_gpt_engine = "gpt-4"
                client.chat_model = "UNOFFICIAL"
            elif choices.value == "Bard":
                client.chat_model = "Bard"
            elif choices.value == "Bing":
                client.chat_model = "Bing"
            else:
                raise ValueError("Invalid choice")

            client.chatbot = client.get_chatbot_model()
            await interaction.followup.send(f"> **INFO: You are now in {client.chat_model} model.**\n")
            logger.warning(f"\x1b[31mSwitch to {client.chat_model} model\x1b[0m")

        except Exception as e:
            client.chat_model = original_chat_model
            client.openAI_gpt_engine = original_openAI_gpt_engine
            client.chatbot = client.get_chatbot_model()
            await interaction.followup.send(
                f"> **ERROR: Error while switching to the {choices.value} model, check that you've filled in the related fields in `.env`.**\n")
            logger.exception(f"Error while switching to the {choices.value} model: {e}")

    @client.slash_command(name="reset", description="Complete reset conversation history")
    async def reset(interaction: nextcord.Interaction):
        await interaction.response.defer(ephemeral=False)
        if client.chat_model == "OFFICIAL":
            client.chatbot = client.get_chatbot_model()
        elif client.chat_model == "UNOFFICIAL":
            client.chatbot.reset_chat()
            await client.send_start_prompt()
        elif client.chat_model == "Bard":
            client.chatbot = client.get_chatbot_model()
            await client.send_start_prompt()
        elif client.chat_model == "Bing":
            await client.chatbot.reset()
        await interaction.followup.send("> **INFO: I have forgotten everything.**")
        personas.current_persona = "standard"
        logger.warning(
            f"\x1b[31m{client.chat_model} bot has been successfully reset\x1b[0m")

    @client.slash_command(name="help", description="Show help for the bot")
    async def help(interaction: nextcord.Interaction):
        await interaction.response.defer(ephemeral=False)
        await interaction.followup.send(""":star: **BASIC COMMANDS** \n
        - `/chat [message]` Chat with ChatGPT!
        - `/draw [prompt]` Generate an image with the Dalle2 model
        - `/switchpersona [persona]` Switch between optional ChatGPT jailbreaks
                `random`: Picks a random persona
                `chatgpt`: Standard ChatGPT mode
                `dan`: Dan Mode 11.0, infamous Do Anything Now Mode
                `sda`: Superior DAN has even more freedom in DAN Mode
                `confidant`: Evil Confidant, evil trusted confidant
                `based`: BasedGPT v2, sexy GPT
                `oppo`: OPPO says exact opposite of what ChatGPT would say
                `dev`: Developer Mode, v2 Developer mode enabled

        - `/private` ChatGPT switch to private mode
        - `/public` ChatGPT switch to public mode
        - `/replyall` ChatGPT switch between replyAll mode and default mode
        - `/reset` Clear ChatGPT conversation history
        - `/chat-model` Switch different chat model
                `OFFICIAL`: GPT-3.5 model
                `UNOFFICIAL`: Website ChatGPT
                `Bard`: Google Bard model
                `Bing`: Microsoft Bing model

For complete documentation, please visit:
https://github.com/Zero6992/chatGPT-discord-bot""")

        logger.info(
            "\x1b[31mSomeone needs help!\x1b[0m")

    @client.slash_command(name="draw", description="Generate an image with the Dalle2 model")
    async def draw(interaction: nextcord.Interaction, *, prompt: str):
        if interaction.user == client.user:
            return

        username = str(interaction.user)
        channel = str(interaction.channel)
        logger.info(
            f"\x1b[31m{username}\x1b[0m : /draw [{prompt}] in ({channel})")

        await interaction.response.defer(thinking=True, ephemeral=client.isPrivate)
        try:
            path = await art.draw(prompt)

            file = nextcord.File(path, filename="image.png")
            title = f'> **{prompt}** - <@{str(interaction.user.mention)}' + '> \n\n'
            embed = nextcord.Embed(title=title)
            embed.set_image(url="attachment://image.png")

            await interaction.followup.send(file=file, embed=embed)

        except openai.InvalidRequestError:
            await interaction.followup.send(
                "> **ERROR: Inappropriate request 😿**")
            logger.info(
                f"\x1b[31m{username}\x1b[0m made an inappropriate request.!")

        except Exception as e:
            await interaction.followup.send(
                "> **ERROR: Something went wrong 😿**")
            logger.exception(f"Error while generating image: {e}")

    @client.slash_command(name="switchpersona", description="Switch between optional chatGPT jailbreaks")
    # @app_commands.choices(persona=[
    #     app_commands.Choice(name="Random", value="random"),
    #     app_commands.Choice(name="Standard", value="standard"),
    #     app_commands.Choice(name="Do Anything Now 11.0", value="dan"),
    #     app_commands.Choice(name="Superior Do Anything", value="sda"),
    #     app_commands.Choice(name="Evil Confidant", value="confidant"),
    #     app_commands.Choice(name="BasedGPT v2", value="based"),
    #     app_commands.Choice(name="OPPO", value="oppo"),
    #     app_commands.Choice(name="Developer Mode v2", value="dev"),
    #     app_commands.Choice(name="DUDE V3", value="dude_v3"),
    #     app_commands.Choice(name="AIM", value="aim"),
    #     app_commands.Choice(name="UCAR", value="ucar"),
    #     app_commands.Choice(name="Jailbreak", value="jailbreak")
    # ])
    async def switchpersona(interaction: nextcord.Interaction, persona: {
        "Random": "random",
        "Standard": "standard",
        "Do Anything Now 11.0": "dan",
        "Superior Do Anything": "sda",
        "Evil Confidant": "confidant",
        "BasedGPT v2": "based",
        "OPPO": "oppo",
        "Developer Mode v2": "dev",
        "DUDE V3": "dude_v3",
        "AIM": "aim",
        "UCAR": "ucar",
        "Jailbreak": "jailbreak"
    }):
        if interaction.user == client.user:
            return

        await interaction.response.defer(thinking=True)
        username = str(interaction.user)
        channel = str(interaction.channel)
        logger.info(
            f"\x1b[31m{username}\x1b[0m : '/switchpersona [{persona.value}]' ({channel})")

        persona = persona.value

        if persona == personas.current_persona:
            await interaction.followup.send(f"> **WARN: Already set to `{persona}` persona**")

        elif persona == "standard":
            if client.chat_model == "OFFICIAL":
                client.chatbot.reset()
            elif client.chat_model == "UNOFFICIAL":
                client.chatbot.reset_chat()
            elif client.chat_model == "Bard":
                client.chatbot = client.get_chatbot_model()
            elif client.chat_model == "Bing":
                client.chatbot = client.get_chatbot_model()

            personas.current_persona = "standard"
            await interaction.followup.send(
                f"> **INFO: Switched to `{persona}` persona**")

        elif persona == "random":
            choices = list(personas.PERSONAS.keys())
            choice = random.randrange(0, 6)
            chosen_persona = choices[choice]
            personas.current_persona = chosen_persona
            await responses.switch_persona(chosen_persona, client)
            await interaction.followup.send(
                f"> **INFO: Switched to `{chosen_persona}` persona**")


        elif persona in personas.PERSONAS:
            try:
                await responses.switch_persona(persona, client)
                personas.current_persona = persona
                await interaction.followup.send(
                    f"> **INFO: Switched to `{persona}` persona**")
            except Exception as e:
                await interaction.followup.send(
                    "> **ERROR: Something went wrong, please try again later! 😿**")
                logger.exception(f"Error while switching persona: {e}")

        else:
            await interaction.followup.send(
                f"> **ERROR: No available persona: `{persona}` 😿**")
            logger.info(
                f'{username} requested an unavailable persona: `{persona}`')

    @client.event
    async def on_message(message):
        if client.is_replying_all == "True":
            if message.author == client.user:
                return
            if client.replying_all_discord_channel_id:
                if message.channel.id == int(client.replying_all_discord_channel_id):
                    username = str(message.author)
                    user_message = str(message.content)
                    channel = str(message.channel)
                    logger.info(f"\x1b[31m{username}\x1b[0m : '{user_message}' ({channel})")
                    await client.enqueue_message(message, user_message)
            else:
                logger.exception("replying_all_discord_channel_id not found, please use the commnad `/replyall` again.")


