# engine/ai_assist.py

from openai import OpenAI
from icecream import ic
import json
import re

class AIAssist:
    def __init__(self, game_manager, world_builder):
        self.game_manager = game_manager
        self.world_builder = world_builder

        # Load API key from engine/secrets.json
        # You can get your API key from https://beta.openai.com/
        secrets_file = 'data/secrets.json'
        with open(secrets_file, 'r') as file:
            secrets = json.load(file)
        self.api_key = secrets.get('openai_api_key', '')

        self.client = OpenAI(api_key=self.api_key)


    def generate_ai_response(self, prompt, max_tokens=250):
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model="gpt-3.5-turbo-1106",  # Change the model based on your requirement
                max_tokens=max_tokens
            )
            print(f"AI generated response: {response.choices[0].message.content.strip()}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            ic(f"Error in AI response generation: {e}")
            return f"I'm having trouble understanding that. {e}"

    def handle_player_command(self, command):
        # Access world data and player sheet
        world_data = self.world_builder.world_data
        player_data = self.game_manager.player_sheet

        # Step 2: Send command to GPT for interpretation
        interpret_prompt = self.construct_command_context_prompt(command, world_data, player_data)
        interpreted_command = self.generate_ai_response(interpret_prompt)


        # Step 3: Process the interpreted command to determine the game action
        action_response = self.determine_action_from_interpretation(interpreted_command)

        # Step 4: Execute the action and generate a narrative response
        return self.generate_narrative_response(action_response, command)
    
    def construct_command_context_prompt(self, command, world_data, player_data):
        # Dictionary of available commands, their descriptions, and examples
        command_dictionary = {
            "move to": {
                "description": "Move the player to a specified location.",
                "examples": ["I want to move to Eldergrove Forest", "go to the Great Bazaar", "travel to the Whispering Forest"],
                "parsing_guideline": "Extract the destination name from the command and format it as '(destination) location'.",
                "command_to_perform": "move to, go to",
                "command_format": "move to <destination>, e.g., go to Eldergrove Forest"
            },
            "take": {
                "description": "Take an item from the current location or container.",
                "examples": ["take potion", "pick up sword", "grab the key"],
                "parsing_guideline": "Identify the item and quantity to be taken from the command.",
                "command_to_perform": "take",
                "command_format": "take <item>, e.g., take potion"
            },
            "examine": {
                "description": "Examine an item or the surroundings.",
                "examples": ["examine map", "look at the statue", "inspect the chest"],
                "parsing_guideline": "Determine the item or area to be examined from the command.",
                "command_to_perform": "examine",
                "command_format": "examine <item>, e.g., examine map"
            },
            "open": {
                "description": "Open a specified container in the current location.",
                "examples": ["open chest", "unlock the door", "open the bag"],
                "parsing_guideline": "Find out which container is to be opened from the command.",
                "command_to_perform": "open",
                "command_format": "open <container>, e.g., open chest"
            },
            "close": {
                "description": "Close the currently open container.",
                "examples": ["close chest", "shut the door", "close"],
                "parsing_guideline": "Understand which container is to be closed, if specified.",
                "command_to_perform": "close",
                "command_format": "close, e.g., close chest"
            },
            "give": {
                "description": "Give an item to an NPC or place it in a container.",
                "examples": ["give potion to guard", "put sword in chest"],
                "parsing_guideline": "Identify the item and recipient (NPC or container) from the command.",
                "command_to_perform": "give",
                "command_format": "give <quantity> <item>, e.g., give 3 health potions"
            },
            "where am i": {
                "description": "Provide the player's current location and description.",
                "examples": ["where am I", "what is this place"],
                "parsing_guideline": "Recognize this as a request for current location information.",
                "command_to_perform": "where am i",
                "command_format": "where am I, e.g., where am I"
            },
            "look around": {
                "description": "Describe the surroundings in the current location.",
                "examples": ["look around", "survey the area", "what do I see"],
                "parsing_guideline": "Interpret as a request to describe the player's current surroundings.",
                "command_to_perform": "look around",
                "command_format": "look around, e.g., look around"
            },
            "fast travel to": {
                "description": "Instantly travel to a different world or major location.",
                "examples": ["fast travel to Shadowlands", "teleport to Mystic Peaks", "I want to fast travel to the Great Bazaar"],
                "parsing_guideline": "Identify the target world or major location for fast travel from the command.",
                "command_to_perform": "fast travel to",
                "command_format": "fast travel to <world/location>, e.g., fast travel to Shadowlands"
            },
            "help": {
                "description": "Display the list of available commands.",
                "examples": ["help", "what can I do"],
                "parsing_guideline": "Acknowledge as a request for available command information.",
                "command_to_perform": "help",
                "command_format": "help, e.g., help"
            }
            # Add other commands here
        }

        # Extract relevant player and world data for context
        current_location = player_data.location.get("location/sublocation", "Unknown Location")
        inventory = player_data.inventory
        current_location_data = self.world_builder.find_location_data(current_location)

        # Add contextual information about the player and the world to the prompt
        player_context = f"Player's current location: '{current_location}'. Player's inventory: {inventory}."
        location_context = f"Current location data: {current_location_data}"

        # Construct the prompt with command context and examples
        prompt = (
            f"As an AI interpreting player commands in a role-playing game, consider the following command dictionary and the player's current situation:\n"
            + "\n".join([
                f"- {cmd}: {desc['description']} (e.g., {', '.join(desc['examples'])}). "
                f"Parsing guideline: {desc['parsing_guideline']}. "
                f"Command to perform: {desc.get('command_to_perform', 'N/A')}. "
                f"Command format: {desc.get('command_format', 'N/A')}"
                for cmd, desc in command_dictionary.items()
            ])
            + "\n\n"
            + player_context + "\n" + location_context + "\n\n"
            f"Player's command: '{command}'\n"
            "Please interpret this command by providing a structured response with 'Action', 'Subject', and 'Quantity' (if applicable). Format the interpretation exactly as shown in these examples:\n"
            "- 'Action: go to, Subject: Great Bazaar'\n"
            "- 'Action: give, Subject: health potion, Quantity: 3'\n\n"
            "Ensure the response follows this format strictly for ease of parsing."
        )
        return prompt
    
    def determine_action_from_interpretation(self, interpreted_command):
        print(f"Determining action from interpreted command: {interpreted_command}")
        command, details, quantity = self.parse_interpreted_command(interpreted_command)
        print(f"Extracted command: {command}, details: {details}, quantity: {quantity}")

        # Directly use the extracted command in your checks
        if command == "move to":
            destination = details  # 'details' holds the subject
            print(f"Extracted destination: {destination}")
            return self.world_builder.move_player(destination) if destination else "Cannot determine destination."

        elif command == "take":
            item_name = details  # 'details' holds the subject
            print(f"Extracted item name: {item_name}")
            return self.world_builder.take_item(item_name) if item_name else "Cannot determine item to take."

        elif command == "examine":
            target = details  # 'details' holds the subject
            print(f"Extracted target: {target}")
            return self.world_builder.examine_item(target) if target else "Cannot determine what to examine."

        elif command == "open":
            container_name = details
            print(f"Extracted container name: {container_name}")
            return self.world_builder.open_container(container_name) if container_name else "Cannot determine container to open."

        elif command == "close":
            print("Command to close container received")
            return self.world_builder.close_container()

        elif command == "give":
            item_info = details  
            print(f"Extracted item info: {item_info}, quantity: {quantity}")
            combined_command = f"{quantity} {item_info}"
            print(f"Combined command: {combined_command}")
            return self.world_builder.give_item(combined_command) if item_info else "Cannot determine item to give."
        
        elif command == "fast travel to":
            world_name = details 
            print(f"Extracted world name for fast travel: {world_name}")
            return self.world_builder.fast_travel_to_world(world_name) if world_name else "Cannot determine destination for fast travel."

        elif command == "where am i":
            print("Command to identify current location received")
            return self.world_builder.where_am_i()

        elif command == "look around":
            print("Command to look around received")
            return self.world_builder.look_around()

        elif command == "help":
            print("Command to display help received")
            return self.world_builder.display_help()

        else:
            print(f"Unknown command received: {command}")
            return "I'm not sure what you're trying to do. Can you phrase it differently?"

    def parse_interpreted_command(self, interpreted_command):
        print(f"Extracting command details from: {interpreted_command}")
        
        # Regular expressions to extract Action, Subject, and Quantity
        # Stops at a comma or a newline
        action_match = re.search(r"Action:\s*([^\n,]+)", interpreted_command)
        subject_match = re.search(r"Subject:\s*([^\n,]+)", interpreted_command)
        quantity_match = re.search(r"Quantity:\s*(\d+)", interpreted_command)

        # Initialize default values
        action = "unknown"
        subject = ""
        quantity = 1  # Default quantity

        # Extract Action
        if action_match:
            action = action_match.group(1).lower().strip()

        # Extract Subject
        if subject_match:
            subject = subject_match.group(1).strip()

        # Extract Quantity
        if quantity_match:
            quantity = int(quantity_match.group(1))

        print(f"Extracted Action: {action}, Subject: {subject}, Quantity: {quantity}")
        return action, subject, quantity

        
    def generate_narrative_response(self, action_response, original_command):
        print(f"Generating narrative response for action: {action_response}, original command: {original_command}")
        player_data = self.game_manager.player_sheet
        current_location = player_data.location.get("location/sublocation", "Unknown Location")
        location_data = self.world_builder.find_location_data(current_location)
        print(f"Current location data for narrative: {location_data}")

        prompt = self.construct_narrative_prompt(action_response, original_command, location_data, current_location)
        print(f"Narrative generation prompt: {prompt}")
        narrative_response = self.generate_ai_response(prompt)
        print(f"Generated narrative response: {narrative_response}")

        return narrative_response

    def construct_narrative_prompt(self, action_response, original_command, location_data, current_location):
        # Extract visible items, features, paths, exits, and containers from the current location data
        visible_items = [item for item in location_data.get('items', []) if item.get('visible', True)]
        visible_features = location_data.get('features', [])
        paths = location_data.get('paths', {})
        exits = [exit for exit in paths.values()]  # Extract exits from paths
        containers = location_data.get('containers', [])

        # Construct a description for each element
        items_description = f"Visible items: {', '.join([item['name'] for item in visible_items])}." if visible_items else "No visible items."
        features_description = f"Visible features: {', '.join(visible_features)}." if visible_features else "No visible features."
        paths_description = f"Available paths: {', '.join(paths.keys())}." if paths else "No visible paths."
        exits_description = f"Exits: {', '.join(exits)}." if exits else "No visible exits."
        containers_description = f"Containers: {', '.join([container['name'] for container in containers])}." if containers else "No visible containers."

        # Construct the prompt for GPT to generate a narrative response
        prompt = (
            f"Based on the following information, generate a narrative response for a role-playing game:\n"
            f"- Player's original command: '{original_command}'\n"
            f"- Action taken: {action_response}\n"
            f"- Current location: {current_location}\n"
            f"- Location description: {location_data.get('description', 'An unknown location.')}\n"
            f"- {items_description}\n"
            f"- {features_description}\n"
            f"- {paths_description}\n"
            f"- {exits_description}\n"
            f"- {containers_description}\n\n"
            "Narrative Response:"
        )
        return prompt
