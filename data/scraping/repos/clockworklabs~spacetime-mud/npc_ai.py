import json
from aiagent_conversation import AIAgentConversation
from autogen.location import Location
from autogen.mobile import Mobile
from autogen.room import Room
from npc_conversation import NPCConversation
from openai_harness import openai_call


class NpcAI:
    prompt_prefix = "You are role-playing as an NPC in an interactive multi-user dungeon (MUD) game. This is a text-based virtual world where players interact with the environment, NPCs, and each other, mainly through text commands and responses. Remember past interactions, understand the current context of the game, and engage players in meaningful interactions. Stay in character, adhere to the game's themes, and provide a rich, immersive experience for the players. \n\n"

    def __init__(self, entity_id, npc_type):
        print("[Behavior] Starting NpcAI with entity_id: " + str(entity_id) + " and npc_type: " + npc_type)
        # Initial properties for the agent
        self.entity_id = entity_id
        self.npc_type = npc_type
        self.conversations = {}
    
    @property
    def getLocation(self):
        return Location.filter_by_spawnable_entity_id(self.entity_id)
    
    @property
    def currentRoom(self):
        """
        Retrieve the current room of the AI from client cache based on entity_id.
        """
        loc = self.getLocation()
        return Room.filter_by_room_id(loc.room_id)

    @property
    def peopleInRoom(self):
        """
        Retrieve a list of people in the current room from client cache.
        """
        # Here, you'd fetch the list of people in the room from your cache
        # e.g., return client_cache.get_people_in_room(self.currentRoom)
        loc = self.getLocation()
        all_locs = Location.filter_by_room_id(loc.room_id)
        valid_mobiles = [loc for loc in all_locs if Mobile.filter_by_entity_id(loc.entity_id) is not None]
        return valid_mobiles

    def evaluateMessage(self, entity_id, message, is_tell):
        """
        Evaluates a message said by a person.
        """

        # only agents respond to tells right now
        if self.npc_type != "ai_agent" and is_tell:
            return

        # If we don't have a conversation with this person, create one.
        if entity_id not in self.conversations:
            if self.npc_type == "ai_agent":
                self.conversations[entity_id] = AIAgentConversation(self.entity_id, entity_id)
            else:
                self.conversations[entity_id] = NPCConversation(self.entity_id, entity_id)
        
        self.conversations[entity_id].message_arrived(message, is_tell)
        
    def enterRoom(self, room):
        """
        Updates the currentRoom and scans for people in the room.
        """
        self.currentRoom = room
        # Scan for people in the room and update self.peopleInRoom
        # ... 

    def leaveRoom(self):
        """
        Moves the AI out of the current room and saves interactions to memory.
        """
        # Save recent interactions to memory
        # ...
        self.currentRoom = None

    def trackEntrance(self, person):
        """
        Track when a person enters the room.
        """
        self.peopleInRoom.append(person)
        # Check memory for past interactions and adjust behavior if needed
        # ...

    def trackExit(self, person):
        """
        Track when a person leaves the room.
        """
        self.peopleInRoom.remove(person)
        # Store recent interactions with the person to memory
        # ...

    def periodicThought(self):
        """
        Periodically generates a statement or action.
        """
        # Generate a random statement or action based on current state
        # or interactions.
        # ...

    def closeConversations(self):
        """
        Closes all conversations.
        """
        for conversation in self.conversations.values():
            conversation.close()