import random
import asyncio
import datetime

import discord
from discord import app_commands

from d20_governance.utils.constants import GOVERNANCE_SVG_ICONS

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from abc import ABC, abstractmethod
from collections import defaultdict
from colorama import Fore, Style


# This is a custom set that tracks order or append and removal as well as groups of channels through sets
class OrderedSet:
    def __init__(self):
        self.set = set()
        self.list = []

    def add(self, item):
        if item not in self.set:
            self.set.add(item)
            self.list.append(item)

    def remove(self, item):
        if item in self.set:
            self.set.discard(item)
            self.list.remove(item)

    def __iter__(self):
        return iter(self.list)

    def __len__(self):
        return len(
            self.list
        )  # The length of the ListSet is the length of the internal list

    def __bool__(self):
        return len(self) > 0  # The instance is "Truthy" if there are elements in it


class RandomCultureModuleManager:
    def __init__(self):
        self.random_culture_module = ""


random_culture_module_manager = RandomCultureModuleManager()


class ValueRevisionManager:
    def __init__(self):
        self.proposed_values_dict = {}
        self.agora_values_dict = {
            "Respect": "Our members should treat each other with respect, recognizing and appreciating diverse perspectives and opinions.",
            "Inclusivity": "Our community strives to be inclusive, creating an environment where everyone feels welcome and valued regardless of their background, identity, or beliefs.",
            "Support": "Our members support and help one another, whether it's providing guidance, advice, or emotional support.",
            "Collaboration": "Our community encourage collaboration, fostering an environment where members can work together and share knowledge or skills.",
            "Trust": "Our community believes building trust is important, as it allows members to feel safe and comfortable sharing their thoughts and experiences.",
        }
        self.selected_value = {}
        self.game_quest_values_dict = {}
        self.quest_game_channels = []
        self.lock = asyncio.Lock()

    def get_value_choices(self):
        choices = [
            app_commands.Choice(name=f"{name}: {value[:60]}", value=name)
            for name, value in value_revision_manager.agora_values_dict.items()
        ]
        return choices

    async def store_proposal(
        self, proposed_value_name_input, proposed_value_definition_input
    ):
        async with self.lock:
            proposed_value_name = proposed_value_name_input.value.strip()
            proposed_value_definition = proposed_value_definition_input.value.strip()
            self.proposed_values_dict[proposed_value_name] = proposed_value_definition

    async def update_values_dict(self, select_value, vote_result):
        async with self.lock:
            if not vote_result:
                print("value dict not updated")
            else:
                if select_value in value_revision_manager.agora_values_dict:
                    del value_revision_manager.agora_values_dict[select_value]
                value_revision_manager.agora_values_dict.update(vote_result)
                message_content = ""
                for (
                    value,
                    description,
                ) in value_revision_manager.agora_values_dict.items():
                    message_content += f"{value}:\n{description}\n\n"
                message = f"```{message_content}```"
                module = CULTURE_MODULES.get("values", None)
                module.config["values_list"] = message

    async def clear_proposed_values(self):
        async with self.lock:
            self.proposed_values_dict.clear()


value_revision_manager = ValueRevisionManager()


class PromptObject:
    def __init__(self):
        self.decision_one = ""
        self.decision_two = ""
        self.decision_three = ""


prompt_object = PromptObject()


class CultureModule(ABC):
    def __init__(self, config):
        self.config = config  # This hold the configuration for the module
        if "guild_channel_map" not in self.config:
            self.config["guild_channel_map"] = {}
        # if "channels" not in self.config:
        # self.config["channels"] = set()

    async def filter_message(
        self, message: discord.Message, message_string: str
    ) -> str:
        return message_string

    # State management with channel and guild mapping
    async def toggle_local_state_per_channel(self, ctx, guild_id, channel_id):
        print("Toggling module...")
        if self.is_local_state_active_in_channel(guild_id, channel_id):
            await self.deactivate_local_state_in_channel(ctx, guild_id, channel_id)
        else:
            await self.activate_local_state_in_channel(ctx, guild_id, channel_id)

    async def toggle_local_state_per_channel(self, ctx, guild_id, channel_id):
        print("Toggling module...")
        if self.is_local_state_active_in_channel(guild_id, channel_id):
            await self.deactivate_local_state_in_channel(ctx, guild_id, channel_id)
        else:
            await self.activate_local_state_in_channel(ctx, guild_id, channel_id)

    def is_local_state_active_in_channel(self, guild_id, channel_id):
        print("Returning local state of active modules...")
        return (
            guild_id in self.config["guild_channel_map"]
            and channel_id in self.config["guild_channel_map"][guild_id]
        )

    async def activate_local_state_in_channel(self, ctx, guild_id, channel_id):
        print("Activating module...")
        if guild_id not in self.config["guild_channel_map"]:
            self.config["guild_channel_map"][guild_id] = set()
        self.config["guild_channel_map"][guild_id].add(channel_id)
        await toggle_culture_module(guild_id, channel_id, self.config["name"], True)
        await display_culture_module_state(
            ctx, guild_id, channel_id, self.config["name"], True
        )

    async def deactivate_local_state_in_channel(self, ctx, guild_id, channel_id):
        print("Deactivating module...")
        channels_for_guild = self.config["guild_channel_map"].get(guild_id, None)
        if channels_for_guild and channel_id in channels_for_guild:
            self.config["guild_channel_map"][guild_id].discard(channel_id)
            await toggle_culture_module(
                guild_id, channel_id, self.config["name"], False
            )
            await display_culture_module_state(
                ctx, guild_id, channel_id, self.config["name"], False
            )

    # Timeout method
    async def timeout(self, ctx, guild_id, channel_id, timeout):
        print("Starting Timeout Task")
        await asyncio.sleep(timeout)
        print("ending Timeout Task")
        if (
            not self.is_local_state_active()
        ):  # Check is module is still deactivated locally after waiting
            # if not
            self.activate_global_state(ctx, guild_id, channel_id)


class Obscurity(CultureModule):
    # Message string may be pre-filtered by other modules
    async def filter_message(
        self, message: discord.Message, message_string: str
    ) -> str:
        print(f"{Fore.GREEN}※ applying obscurity module{Style.RESET_ALL}")

        # Get the method from the module based on the value of "mode"
        method = getattr(self, self.config["mode"])

        # Call the method
        filtered_message = method(message_string)
        return filtered_message

    def scramble(self, message_string):
        words = message_string.split()
        scrambled_words = []
        for word in words:
            if len(word) <= 3:
                scrambled_words.append(word)
            else:
                middle = list(word[1:-1])
                random.shuffle(middle)
                scrambled_words.append(word[0] + "".join(middle) + word[-1])
        return " ".join(scrambled_words)

    def replace_vowels(self, message_string):
        vowels = "aeiou"
        message_content = message_string.lower()
        return "".join([" " if c in vowels else c for c in message_content])

    def pig_latin(self, message_string):
        words = message_string.split()
        pig_latin_words = []
        for word in words:
            if word[0] in "aeiouAEIOU":
                pig_latin_words.append(word + "yay")
            else:
                first_consonant_cluster = ""
                rest_of_word = word
                for letter in word:
                    if letter not in "aeiouAEIOU":
                        first_consonant_cluster += letter
                        rest_of_word = rest_of_word[1:]
                    else:
                        break
                pig_latin_words.append(rest_of_word + first_consonant_cluster + "ay")
        return " ".join(pig_latin_words)

    def camel_case(self, message_string):
        words = message_string.split()
        camel_case_words = [word.capitalize() for word in words]
        return "".join(camel_case_words)


class Wildcard(CultureModule):
    async def filter_message(
        self, message: discord.Message, message_string: str
    ) -> str:
        """
        A LLM filter for messages made by users
        """
        print(f"{Fore.GREEN}※ applying wildcard module{Style.RESET_ALL}")
        module = CULTURE_MODULES.get("wildcard", None)
        llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")
        prompt = PromptTemplate(
            input_variables=[
                "input_text",
                "group_name",
                "group_topic",
                "group_way_of_speaking",
            ],
            template="You are from {group_name}. Please rewrite the following input ina way that makes the speaker sound {group_way_of_speaking} while maintaining the original meaning and intent. Incorporate the theme of {group_topic}. Don't complete any sentences, just rewrite them. Input: {input_text}",
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = await chain.arun(
            {
                "group_name": prompt_object.decision_one,
                "group_topic": prompt_object.decision_two,
                "group_way_of_speaking": prompt_object.decision_three,
                "input_text": message_string,
            }
        )
        return response


class Amplify(CultureModule):
    async def filter_message(
        self, message: discord.Message, message_string: str
    ) -> str:
        """
        A LLM filter for messages during the /eloquence command/function
        """
        print(f"{Fore.GREEN}※ applying amplify module{Style.RESET_ALL}")
        llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")
        prompt = PromptTemplate(
            input_variables=["input_text"],
            template="Using the provided input text, generate a revised version that amplifies its sentiment to a much greater degree. Maintain the overall context and meaning of the message while significantly heightening the emotional tone. You must ONLY respond with the revised message. Input text: {input_text}",
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = await chain.arun(message_string)
        return response


class Ritual(CultureModule):
    async def filter_message(
        self, message: discord.Message, message_string: str
    ) -> str:
        print(f"{Fore.GREEN}※ applying ritual module{Style.RESET_ALL}")
        async for msg in message.channel.history(limit=100):
            if msg.id == message.id:
                continue
            if msg.author.bot and not msg.content.startswith(
                "※"
            ):  # This condition lets webhook messages to be checked
                continue
            if msg.content.startswith("/") or msg.content.startswith("-"):
                continue
            previous_message = msg.content
            break
        if previous_message is None:
            return message_string
        filtered_message = await self.initialize_ritual_agreement(
            previous_message, message_string
        )
        return filtered_message

    async def initialize_ritual_agreement(self, previous_message, new_message):
        llm = ChatOpenAI(temperature=0.9)
        prompt = PromptTemplate(
            input_variables=["previous_message", "new_message"],
            template="Write a message that reflects the content in the message '{new_message}' but is cast in agreement with the message '{previous_message}'. Preserve and transfer the meaning and any spelling errors or text transformations in the message in the response.",
        )  # FIXME: This template does not preserve obscurity text processing. Maybe obscurity should be reaplied after ritual if active in the active_culture_mode list
        chain = LLMChain(llm=llm, prompt=prompt)
        response = await chain.arun(
            previous_message=previous_message, new_message=new_message
        )
        return response


class Values(CultureModule):
    async def check_values(self, bot, ctx, message: discord.Message):
        print("Checking values")
        if message.reference:
            reference_message = await message.channel.fetch_message(
                message.reference.message_id
            )
            if (
                reference_message.author.bot
                and not reference_message.content.startswith(
                    "※"
                )  # This condition lets webhook messages to be checked
            ):
                await ctx.send("Cannot check values of messages from bot")
                return
            else:
                print(
                    f"Original Message Content: {reference_message.content}, posted by {message.author}"
                )

            current_values_dict = value_revision_manager.agora_values_dict
            values_list = f"Community Defined Values:\n\n"
            for value in current_values_dict.keys():
                values_list += f"* {value}\n"
            llm_response, alignment = await self.llm_analyze_values(
                current_values_dict, reference_message.content
            )
            message_content = f"----------```Message: {reference_message.content}\n\nMessage author: {reference_message.author}```\n> **Values Analysis:** {llm_response}\n```{values_list}```\n----------"

            # Assign alignment roles to users if their post is values-checked
            if alignment == "aligned":
                await assign_role_to_user(message.author, "Aligned")
            else:
                await assign_role_to_user(message.author, "Misaligned")
            await ctx.send(message_content)
        else:
            if message.author.bot and not message.content.startswith("※"):
                await ctx.send("Cannot check values of messages from bot")
                return
            else:
                print(
                    f"Original Message Contnet: {message.content}, posted by {message.author}"
                )

            current_values_dict = value_revision_manager.agora_values_dict
            values_list = f"Community Defined Values:\n\n"
            for value in current_values_dict.keys():
                values_list += f"* {value}\n"
            llm_response, alignment = await self.llm_analyze_values(
                current_values_dict, message.content
            )
            message_content = f"----------```Message: {message.content}\n\nMessage author: {message.author}```\n> **Values Analysis:** {llm_response}\n```{values_list}```\n----------"

            # Assign alignment roles to users if their post is values-checked
            if alignment == "aligned":
                await assign_role_to_user(message.author, "Aligned")
            else:
                await assign_role_to_user(message.author, "Misaligned")
            await ctx.send(message_content)

    async def llm_analyze_values(self, values_dict, text):
        """
        Analyze message content based on values
        """
        print(f"{Fore.GREEN}※ applying values module{Style.RESET_ALL}")
        llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
        template = f"We hold and maintain a set of mutually agreed-upon values. Analyze whether the message '{text}' is in accordance with the values we hold:\n\n"
        current_values_dict = value_revision_manager.agora_values_dict
        for (
            value,
            description,
        ) in values_dict.items():
            template += f"- {value}: {description}\n"
        template += f"\nNow, analyze the message:\n{text}. Start the message with either the string 'This message aligns with our values' or 'This message does not align with our values'. Then briefly explain why the values are aligned or misaligned based on the values the group holds. Use no more than 250 characters."
        prompt = PromptTemplate.from_template(template=template)
        chain = LLMChain(llm=llm, prompt=prompt)
        response = await chain.arun({"text": text})
        alignment = (
            "aligned"
            if "This message aligns with our values" in response
            else "misaligned"
        )
        return response, alignment

    # TODO: Finish implementing and refine
    # async def randomly_check_values(self, bot, ctx, channel):
    #     while True:
    #         print("in a value check loop")
    #         current_time = datetime.datetime.utcnow()
    #         print(f"time is: {current_time}")
    #         # Randomly generate delay between executions
    #         delay = random.randint(45, 55)

    #         # Wait for the specified delay
    #         await asyncio.sleep(delay)

    #         try:
    #             # Fetch a random message from a game channel
    #             messages = []
    #             async for message in channel.history(limit=100, after=current_time):
    #                 if (
    #                     message.content.startswith("※")
    #                     or isinstance(message, discord.Message)
    #                     and not message.author.bot
    #                 ):
    #                     messages.append(message)

    #             if not messages:
    #                 print("No valid messages found in the channel")
    #                 return

    #             # Generate a list of valid message IDs
    #             valid_message_ids = [message.id for message in messages]

    #             random_message_id = random.choice(valid_message_ids)

    #             random_message = await channel.fetch_message(random_message_id)
    #             print("fetched random message")

    #             # Check values of the random message
    #             await self.check_values(bot, channel, random_message)
    #         except Exception as e:
    #             print(f"Error occurred while checking values: {e}")
    #         except discord.NotFound:
    #             print("Random message not found")
    #         except discord.HTTPException as e:
    #             print(f"Error occurrent while fetching random message: {e}")


async def assign_role_to_user(user, role_name):
    guild = user.guild
    new_role = discord.utils.get(guild.roles, name=role_name)

    # Define role variables
    aligned_role = discord.utils.get(guild.roles, name="Aligned")
    misaligned_role = discord.utils.get(guild.roles, name="Misaligned")

    # Check if the user already has an alignment role
    if aligned_role in user.roles and new_role == misaligned_role:
        await user.remove_roles(aligned_role)
    elif misaligned_role in user.roles and new_role == aligned_role:
        await user.remove_roles(misaligned_role)

    await user.add_roles(new_role)


class Eloquence(CultureModule):
    async def filter_message(
        self, message: discord.Message, message_string: str
    ) -> str:
        """
        A LLM filter for messages during the /eloquence command/function
        """
        print(f"{Fore.GREEN}※ applying eloquence module{Style.RESET_ALL}")
        llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
        prompt = PromptTemplate.from_template(
            template="You are from the Shakespearean era. Please rewrite the following input in a way that makes the speaker sound as eloquent, persuasive, and rhetorical as possible, while maintaining the original meaning and intent. Don't complete any sentences, jFust rewrite them. Input: {input_text}"
        )
        prompt.format(
            input_text=message_string
        )  # TODO: is both formatting and passing the message_string necessary?
        chain = LLMChain(llm=llm, prompt=prompt)
        response = await chain.arun(message_string)
        return response


ACTIVE_MODULES_BY_CHANNEL = defaultdict(OrderedSet)


async def toggle_culture_module(guild_id, channel_id, module_name, state):
    """
    If state is True, turn on the culture module
    if state is False, turn off the culture module
    """
    key = (guild_id, channel_id)
    active_modules_by_channel = ACTIVE_MODULES_BY_CHANNEL[key]

    if state:
        active_modules_by_channel.add(module_name)
    else:
        active_modules_by_channel.remove(module_name)


# TODO: what does the variable "state" mean?
async def display_culture_module_state(ctx, guild_id, channel_id, module_name, state):
    """
    Send an embed displaying state of active culture moduled by channel
    """
    print("Displaying culture module state...")
    key = (guild_id, channel_id)
    active_modules_by_channel = ACTIVE_MODULES_BY_CHANNEL[key]

    module = CULTURE_MODULES[module_name]

    # TODO: make state a more descriptive variable name
    if state:
        name = "Activated"
        value = module.config["activated_message"]
    else:
        name = "Deactivated"
        value = module.config["deactivated_message"]

    if active_modules_by_channel.list:
        active_culture_module_values = ", ".join(active_modules_by_channel.list)
    else:
        active_culture_module_values = "none"

    embed = discord.Embed(
        title=f"Culture: {module_name.upper()}", color=discord.Color.dark_gold()
    )
    embed.set_thumbnail(url=module.config["url"])
    embed.add_field(
        name=name,
        value=value,
        inline=False,
    )
    if module.config["mode"] is not None and state:
        embed.add_field(
            name="Mode:",
            value=module.config["mode"],
            inline=False,
        )
    if module.config["message_alter_mode"] == "llm" and state:
        embed.add_field(
            name="LLM Prompt:",
            value=module.config["llm_disclosure"],
            inline=False,
        )
    if module.config["help"] and state:
        embed.add_field(
            name="How to use:",
            value=module.config["how_to_use"],
            inline=False,
        )
    embed.add_field(
        name="Active Culture Modules:",
        value=active_culture_module_values,
        inline=False,
    )
    if module.config["values_list"] is not None and state:
        embed.add_field(
            name="List of Current Community Values:",
            value=module.config["values_list"],
            inline=False,
        )

    await ctx.send(embed=embed)


def get_values_message():
    message_content = ""
    for (
        value,
        description,
    ) in value_revision_manager.agora_values_dict.items():
        message_content += f"{value}:\n{description}\n\n"
    message = f"```{message_content}```"
    return message


values_list = get_values_message()

CULTURE_MODULES = {
    "wildcard": Wildcard(
        {
            "name": "wildcard",
            "global_state": False,
            "local_state": False,
            "mode": None,
            "help": False,
            "message_alter_mode": "llm",
            "llm_disclosure": None,
            "activated_message": "Messages will now be process through an LLM.",
            "deactivated_message": "Messages will no longer be processed through an LLM.",
            "url": "",  # TODO: Add Wildcard URL
            "icon": GOVERNANCE_SVG_ICONS["culture"],
            "input_value": 0,
            "values_list": None,
        }
    ),
    "obscurity": Obscurity(
        {
            "name": "obscurity",
            "global_state": False,
            "local_state": False,
            "mode": "scramble",
            "help": False,
            "message_alter_mode": "text",
            "alter_message": True,
            "activated_message": "Messages will be distored based on mode of obscurity.",
            "deactivated_message": "Messages will no longer be distored by obscurity.",
            "url": "https://raw.githubusercontent.com/metagov/d20-governance/main/assets/imgs/embed_thumbnails/obscurity.png",
            "icon": GOVERNANCE_SVG_ICONS["culture"],
            "input_value": 0,
            "values_list": None,
        }
    ),
    "eloquence": Eloquence(
        {
            "name": "eloquence",
            "global_state": False,
            "local_state": False,
            "mode": None,
            "help": False,
            "message_alter_mode": "llm",
            "llm_disclosure": "You are from the Shakespearean era. Please rewrite the messages in a way that makes the speaker sound as eloquent, persuasive, and rhetorical as possible, while maintaining the original meaning and intent.",
            "activated_message": "Messages will now be process through an LLM.",
            "deactivated_message": "Messages will no longer be processed through an LLM.",
            "url": "https://raw.githubusercontent.com/metagov/d20-governance/main/assets/imgs/embed_thumbnails/eloquence.png",
            "icon": GOVERNANCE_SVG_ICONS["culture"],
            "input_value": 0,
            "values_list": None,
        }
    ),
    "ritual": Ritual(
        {
            "name": "ritual",
            "global_state": False,
            "local_state": False,
            "mode": None,
            "help": False,
            "message_alter_mode": "llm",
            "llm_disclosure": "Write a message that reflects the content in the posted message and is cast in agreement with the previous message. Preserve and transfer any spelling errors or text transformations in these messages in the response.",
            "activated_message": "A ritual of agreement permeates throughout the group.",
            "deactivated_message": "Automatic agreement has ended. But will the effects linger in practice?",
            "url": "",  # TODO: make ritual img
            "icon": GOVERNANCE_SVG_ICONS["culture"],
            "input_value": 0,
            "values_list": None,
        }
    ),
    "amplify": Amplify(
        {
            "name": "amplify",
            "global_state": False,
            "local_state": False,
            "mode": None,
            "help": False,
            "message_alter_mode": "llm",
            "llm_disclosure": "Using the provided input text, generate a revised version that amplifies its sentiment to a much greater degree. Maintain the overall context and meaning of the message while significantly heightening the emotional tone.",
            "activated_message": "Sentiment amplification abounds.",
            "deactivated_message": "Sentiment amplification has ceased.",
            "url": "",  # TODO: make amplify img
            "icon": GOVERNANCE_SVG_ICONS["culture"],
            "input_value": 0,
            "values_list": None,
        }
    ),
    "values": Values(
        {
            "name": "values",
            "global_state": False,
            "mode": None,
            "help": True,
            "how_to_use": "Your posts are now subject to alignment analysis. The content of posts will be randomly analyized to see how aligned they are with the group's values. You can also reply to the message you want to check and type `check-values`. you will be labled either aligned or misaligned based on analysis.",
            "local_state": False,
            "message_alter_mode": None,
            "llm_disclosure": "You hold and maintain a set of mutually agreed upon values. The values you maintain are the values defined by the community. You review the contents of messages sent for validation and analyze the contents in terms of the values you hold. You describe in what ways the input text are aligned or unaligned with the values you hold.",
            "activated_message": "A means of validating the cultural alignment of this online communiuty is nafculture_moduow available. Respond to a message with check-values.",
            "deactivated_message": "Automatic measurement of values is no longer present, through an essence of the culture remains, and you can respond to messages with `check-values` to check value alignment.",
            "url": "",  # TODO: make values img
            "icon": GOVERNANCE_SVG_ICONS["culture"],
            "input_value": 0,
            "values_list": values_list,
        }
    ),
}


async def send_msg_to_random_player(game_channel):
    print("Sending random DM...")
    players = [member for member in game_channel.members if not member.bot]
    random_player = random.choice(players)
    dm_channel = await random_player.create_dm()
    await dm_channel.send(
        "🌟 Greetings, esteemed adventurer! A mischievous gnome has entrusted me with a cryptic message just for you: 'In the land of swirling colors, where unicorns prance and dragons snooze, a hidden treasure awaits those who dare to yawn beneath the crescent moon.' Keep this message close to your heart and let it guide you on your journey through the wondrous realms of the unknown. Farewell, and may your path be ever sprinkled with stardust! ✨"
    )
