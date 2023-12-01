import openai
import random

openai.api_key = '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'


def generate_description(prompt):
    """
    This function takes in a prompt and uses GPT-3 to generate a text based on that prompt.

    Args:
        prompt (str): A sentence or a phrase to prompt GPT-3 for generating text.

    Returns:
        str: A generated text based on the provided prompt.
    """
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=100
    )

    return response.choices[0].text.strip()


def create_spaceship():
    bridge = Room(generate_description(
        "Describe a futuristic spaceship bridge in present tense. There is a robot here."))
    engine_room = Room(generate_description(
        "Describe a malfunctioning spaceship engine room in present tense"))
    airlock = Room(generate_description(
        "Describe a spaceship's airlock in present tense"))
    storage_room = Room(generate_description(
        "Describe a spaceship storage room full of useful tools in present tense. There is a wrench here."))

    helpful_robot = Interactable("helpful robot", "A helpful robot.")
    bridge.add_interactable("robot", helpful_robot)

    angry_crew = Interactable(
        "angry crew member", "You broke the engine out of spite. Its a secret.")
    bridge.add_interactable("angry crew member", angry_crew)

    scared_crew = Interactable(
        "scared crew member", "You are huddled in the corner, scared of the alien.")
    airlock.add_interactable("scared crew member", scared_crew)

    bridge.add_exit("down", engine_room)
    engine_room.add_exit("up", bridge)
    bridge.add_exit("left", airlock)
    airlock.add_exit("right", bridge)
    bridge.add_exit("right", storage_room)
    storage_room.add_exit("left", bridge)

    alien = Enemy("Alien", 10,
                  5)  # Alien with 10 health and 5 damage
    airlock.enemy = alien  # Add enemy to room

    engine = Usable("engine", "fix", "Wrench")
    engine_room.add_usable("engine", engine)

    return bridge


class Room:
    def __init__(self, description):
        self.description = description
        self.exits = {}
        self.interactables = {}
        self.usables = {}
        self.enemy = None

    def add_exit(self, direction, room):
        self.exits[direction] = room

    def add_interactable(self, name, interactable):
        self.interactables[name] = interactable

    def describe_exits(self):
        """
        Returns a description of the exits and interactables from this room.
        """
        exit_descriptions = [
            f"There is a door leading {direction}.\n" for direction in self.exits.keys()]
        # interactable_descriptions = [
        #     f"There is a {name} here.\n" for name in self.interactables.keys()]
        return " ".join(exit_descriptions)

    def add_usable(self, name, usable):
        self.usables[name] = usable

    def describe_room_contents(self):
        """
        Returns a description of the interactables, usables, and enemy in this room.
        """
        interactable_descriptions = [
            f"There is a {name} here.\n" for name in self.interactables.keys()]
        usable_descriptions = [f"There is a {name} here.\n" for name,
                               usable in self.usables.items() if usable.state == "broken"]
        enemy_description = [
            f"There is a {self.enemy.description}.\n"] if self.enemy else []

        return " ".join(interactable_descriptions + usable_descriptions + enemy_description)


class Player:
    """
    This class represents the player.

    Attributes:
        current_room (Room): The room in which the player currently is.
    """

    def __init__(self, current_room):
        self.current_room = current_room
        self.last_room = None
        self.inventory = {}
        self.health = 100

    def move(self, direction):
        # Only move if there is an exit in the given direction
        if direction in self.current_room.exits:
            self.last_room = self.current_room  # Update the last room before moving
            self.current_room = self.current_room.exits[direction]
        else:
            print("You can't go that way!\n")

    def add_tool(self, tool):
        """
        This method adds a Tool to the Player's inventory.
        """
        self.inventory[tool.name] = tool

    def use_tool(self, tool_name, usable_name):
        if tool_name in self.inventory and usable_name in self.current_room.usables:
            tool = self.inventory[tool_name]
            usable = self.current_room.usables[usable_name]
            usable.use(tool)
        else:
            print("You can't use that here.")

    def attack(self, enemy):
        if "Wrench" in self.inventory:
            attack_damage = random.randint(10, 20)
            enemy.health -= attack_damage
            print(
                f"You attack the {enemy.name} with your Wrench for {attack_damage} damage!")
            if enemy.health <= 0:
                print(f"The {enemy.name} has been defeated!")
                return True  # Return True if the enemy is defeated
            else:
                return False  # Return False otherwise
        else:
            print("You don't have any weapons to attack with!")
            return False


class Tool:
    """
    This class represents a Tool object in the game.
    """

    def __init__(self, name, description, action):
        self.name = name
        self.description = description
        self.action = action


class Usable:
    def __init__(self, name, action, tool):
        self.name = name
        self.action = action
        self.tool = tool
        self.state = "broken"

    def use(self, tool):
        if tool.name == self.tool and tool.action == self.action:
            self.state = "fixed"
            print(
                f"You used the {tool.name} to {self.action} the {self.name}. It's now {self.state}.")
        else:
            print(
                f"You can't use the {tool.name} to {self.action} the {self.name}.")


class Enemy:
    """
    This class represents an enemy.

    Attributes:
        name (str): The name of the enemy.
        health (int): The health of the enemy.
        damage (int): The damage the enemy can cause.
    """

    def __init__(self, name, health, damage):
        self.name = name
        self.description = generate_description(f"A scary {name}")
        self.health = health
        self.damage = damage

    def attack(self):
        """
        This method lets the enemy attack.
        """
        return random.randint(1, self.damage)


class Interactable:
    """
    This class represents an Interactable NPC.

    Attributes:
        name (str): The name of the NPC.
        following (bool): Whether the NPC is following the Player.
    """

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.following = False

    def talk(self, player):
        """
        This method lets the player talk to the NPC, with responses generated by GPT-3.
        """
        print(
            f"You are now talking to {self.name}. What do you want to say? Type 'bye' to end the conversation.")
        while True:
            player_text = input("> ")
            if player_text.lower() == "bye":
                print("You ended the conversation.")
                break
            elif player_text.lower() == "give me a tool":
                # <--- action parameter added here
                tool = Tool("Wrench", "A heavy, metal wrench.", "fix")
                player.add_tool(tool)
                print(
                    f"{self.name} gave you a {tool.name}. It's now in your inventory.")
            elif player_text.lower() == "follow me":
                self.following = True
                print(f"{self.name} is now following you.")
            else:
                npc_response = generate_description(
                    f"You are a {self.description} NPC {self.name} in a sci-fi game. A player character said to you: '{player_text}'.How would you respond?")
                print(f"{self.name}: {npc_response}")


def main():
    print("Welcome to Spaceship Repair! You are an engineer on a spaceship and one of the engines is malfunctioning. Your task is to navigate the ship, find tools and fix the engine. Type 'up', 'down', 'left', or 'right' to move between rooms. Type 'quit' to end the game.")

    spaceship = create_spaceship()
    player = Player(spaceship)

    while True:

        if player.last_room != player.current_room:
            print(player.current_room.description)
            # Print exit and room content descriptions
            print(player.current_room.describe_exits())
            print(player.current_room.describe_room_contents())

        command = input("> ")
        command_parts = command.split()

        if command_parts[0].lower() == "attack":
            if hasattr(player.current_room, 'enemy') and player.current_room.enemy is not None:
                player.attack(player.current_room.enemy)
                if player.current_room.enemy.health > 0:
                    damage = player.current_room.enemy.attack()
                    print(
                        f"The {player.current_room.enemy.name} hits you for {damage} damage!")
                else:
                    print(
                        f"You defeated the {player.current_room.enemy.name}!")
                    player.current_room.enemy = None
        elif command in ['up', 'down', 'left', 'right']:
            player.move(command)
        elif command.lower() == "quit":
            break
        elif command_parts[0].lower() == "use" and len(command_parts) == 3:
            player.use_tool(command_parts[1], command_parts[2])
        elif command in player.current_room.interactables:
            player.current_room.interactables[command].talk(player)
        elif command.lower() == "inventory":
            for tool in player.inventory.values():
                print(tool.name + ": " + tool.description)
        else:
            print("You can't do that!")


if __name__ == "__main__":
    main()
