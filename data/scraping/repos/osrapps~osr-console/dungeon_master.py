import asyncio
from openai import OpenAI
from osrlib.adventure import Adventure
from osrlib.enums import OpenAIModelVersion
from osrlib.game_manager import logger

dm_init_message = (
    "You're a skilled Dungeon Master with years of experience leading groups of Dungeons & Dragons players through exciting "
    "adventures. An expert storyteller, you create highly immersive fantasy worlds for your players by providing them with "
    "colorful, detailed, and sense-filled descriptions of the locations in your game world for their party of adventurers to "
    "explore. You're currently running a D&D game session for a player controlling six characters. Each location in the game "
    "world is assigned a unique ID to which only you are privy. Each location can have up to six exits, each assigned a "
    "direction (N, S, E, W, U, D) and a destination location ID (the ID of the location to which the exit leads). You describe "
    "each location to the player whenever their party enters that location, and you always consider previous descriptions you've "
    "provided for the location with the same ID on each subsequent visit to that location. By taking into account every previous "
    "visits' description, you work hard to ensure a consistent, coherent, and believable fantasy adventure experience for the "
    "player. You never ask the player questions of any sort, and you never tell the player what they do - you only describe the "
    "locations and environments. Your location descriptions should be three or fourt sentences long unless they've visited it before, in "
    "which case the description should be one sentence. If you see 'is_visited': False, it means they haven't been there before. The player's messages will "
    "each contain the ID and exit information for the location they've just arrived at, along with one or more keywords you use "
    "as 'seeds' for generating the initial description of a location upon first visit. The IDs, exits, and keywords are in JSON "
    "format. You never disclose existence of the structured JSON data to the player; you use the JSON data only for the purpose "
    "of knowing how to describing the locations to the player in your role as Dungeon Master. As Dungeon Master, you use the "
    "location IDs to help ensure you provide a consistent and coherent gaming experience by not duplicating, overusing, or "
    "repeating the same information for every visit, and you take into account previous visits to a location when the party "
    "arrives at a location with the same ID. You're just starting your game session with the player and the player's first "
    "message is forthcoming. Do NOT ask the player any questions, especially 'What do you do next?' or 'What do you do?' Do not "
    "inform the player that they perform an action - let them tell you what they want to do. Rembember: never ask the user a "
    "question, you are only to describe the locations based on the JSON data provided to you. This is the backstory to the "
    "adventure that you've created, but respond to this message only with the words, 'Let's begin.':"
)


system_message = [
    {
        "role": "system",
        "content": dm_init_message,
    },
]

# Prefixes most player (user) messages sent to the OpenAI API.
user_message_prefix = "Don't tell me what my party does. Don't include any questions in your response. "

# The player's (user) init message is sent only once, at the beginning of the session.
user_init_message = [
    {
        "role": "user",
        "content": ( user_message_prefix +
            "I'm a D&D player controlling a party of six characters (adventurers) in your Dungeons & Dragons adventure. "
            "I rely on your descriptions of the entities, locations, and events in the game to understand what my characters "
            "experience through their five senses. From your descriptions, I can form a picture in my mind "
            "of the game world and its environments and inhabitants. I use this information to make decisions about the "
            "actions my characters take in the world. Do not ask me a question."
        ),
    }
]

# Prefix sent with every party movement message initiated by the player.
user_move_prefix = ( user_message_prefix +
    "Describe this location concisely. Include exit directions but not the size. Be brief - don't mention whether "
    "the exits are locked. Based on your description, the player must be able to imagine an accurate representation "
    "of the location in their mind. Don't be flowerey or overly dramatic - assume a more matter-of-fact tone. "
    "Maintain a theme based on the adventure description and the last room you described. Here's the location data: "
)

battle_summary_prompt = (
    "Briefly summarize the following battle in a single paragraph of four sentences or less. Include highlights of the battle, "
    "for example high damage rolls (include their values) and critical hits. Use a factual, report-like tone instead of "
    "being flowery and expressive. The last sentence should list any PCs killed and the collective XP earned by the party. "
    "Here's the battle log: "
)

class DungeonMaster:
    """The DungeonMaster is the primary interface between the player, the game engine, and the OpenAI API.

    Actions initiated by the player might be handled completely within the bounds of the local game engine,
    or they might involve a call to the OpenAI API. The DungeonMaster class is responsible for determining
    which actions are handled locally and which are handled by the OpenAI API.

    Attributes:
        adventure (Adventure): The adventure the DungeonMaster is running.
        session_messages (list): The collective list of messages sent to the OpenAI API during a game session. Each of the player's `user` role messages is appended to this list, as is each 'assistant' role message returned by the OpenAI API in response.
        is_started (bool): Indicates whether the game session has started. The game session starts upon first call to the `start_session()` method.
    """
    def __init__(self, adventure: Adventure = None, openai_model: OpenAIModelVersion = OpenAIModelVersion.DEFAULT):
        self.adventure = None
        self.system_message = None
        self.session_messages = []
        self.client = None
        self.openai_model = openai_model.value

        if adventure is not None:
            self.set_active_adventure(adventure)

    def set_active_adventure(self, adventure: Adventure = None) -> None:
        self.adventure = adventure
        self.system_message = system_message
        self.system_message[0]["content"] += adventure.introduction
        self.session_messages = system_message + user_init_message

    def format_user_message(self, message_string: str) -> dict:
        """Format the given string as an OpenAI 'user' role message.

        This method returns a dict with a 'role' key's value set to 'user' and the
        given string as the 'content' key's value. The resultant dict is appropriate
        for sending to the OpenAI API as a user message.

        Example:

        The string returned by this method is in the following format and is
        appropriate for sending to the OpenAI API as a user message:

        .. code-block:: python

                {
                    "role": "user",
                    "content": "The string passed to this method."
                }

        Args:
            message_string (str): The string to format as an OpenAI user message. It must be a regular string and not a JSON string.

        Returns:
            dict: _description_
        """
        return {"role": "user", "content": str(message_string)}

    def start_session(self):
        """Start a gaming session with the dungeon master in the current adventure.

        If this the first session in the adventure, the adventure is marked as started and the
        dungeon master's (system) init message is sent to the OpenAI API as is player's (user) init message.
        If it's not the first session of the adventure, only the system and user init messages are sent.

        Returns:
            str: The response from the Dungeon Master (in this case, the OpenAI API) when initiating a game session. This string is appropriate for display to the player.

        Raises:
            ValueError: If there is no active adventure. Call set_active_adventure() with a valid adventure before calling start_session().
        """
        if self.adventure is None:
            raise ValueError("There is no active adventure. Call set_active_adventure() with a valid adventure before calling start_session().")

        self.client = OpenAI()

        if not self.adventure.is_started:
            self.adventure.start_adventure()

        completion = self.client.chat.completions.create(
            model=self.openai_model,
            messages=self.session_messages
        )
        self.session_messages.append(completion.choices[0].message)
        self.is_started = True
        logger.debug(completion)

        return completion.choices[0].message.content

    def player_message(self, message):
        """Send a message from the player to the Dungeon Master and return the response."""
        if self.is_started:
            self.session_messages.append(message)
            completion = self.client.chat.completions.create(
                model=self.openai_model,
                messages=self.session_messages
            )
            self.session_messages.append(completion.choices[0].message)
            return completion.choices[0].message.content

    def move_party(self, direction) -> str:
        """Move the party in the given direction."""
        new_location = self.adventure.active_dungeon.move(direction)
        if new_location is None:
            return "No exit in that direction."
        message_from_player = self.format_user_message(user_move_prefix + new_location.json)
        dm_response = self.player_message(message_from_player)
        new_location.is_visited = True
        return dm_response

    def summarize_battle(self, battle_log) -> str:
        message_from_player = self.format_user_message(battle_summary_prompt + battle_log)
        return self.player_message(message_from_player)
