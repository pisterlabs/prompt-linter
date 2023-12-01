import random
import pickle
import openai

class Game:
    EVENTS = [
        "You find a treasure chest.",
        "You meet a friendly stranger.",
        "You are transported to a different world."
    ]

    def __init__(self) -> None:
        self.client = openai.Client(api_key="YOUR_API_KEY_HERE")
        self.player = None
        self.locations = [
            Location("Start Location", "You find yourself in a mysterious place."),
            Location("Forest", "A dense forest with towering trees.", Enemy("Wolf", 30, 5, 2)),
            Location("Cave", "A dark and damp cave.", Enemy("Goblin", 50, 10, 5))
        ]
        self.current_location = self.locations[0]
        self.story = ""

    def initialize(self) -> None:
        self.show_main_menu()

    def show_main_menu(self) -> None:
        print("=== Main Menu ===")
        print("1. New Game")
        print("2. Load Game")
        print("3. Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            self.start_new_game()
        elif choice == "2":
            self.load_game()
        elif == "3":
            print("Thanks for playing!")
            return
        else:
            print("Invalid choice. Please try again.")
            self.show_main_menu()

    def start_new_game(self) -> None:
        print("=== Character Creation ===")
        name = input("Enter your character's name: ")
        character_class = input("Choose a character class (Warrior/Mage/Rogue): ")
        self.player = Player(name, character_class)
        self.current_location = self.locations[0]
        self.story = ""
        print("Character created. Please choose 'Start Game' to start playing!")

    def save_game(self) -> None:
        with open('savegame.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        print("Game saved successfully.")

    def load_game(self) -> None:
        try:
            with open('savegame.pkl', 'rb') as f:
                loaded_game = pickle.load(f)
                self.__dict__.update(loaded_game.__dict__)
            print("Game loaded successfully.")
        except FileNotFoundError:
            print("No saved game found.")

    def play_game(self) -> None:
        if self.player is None:
            print("No character found. Please create a new character first.")
            return
        print(f"=== {self.current_location.name} ===")
        print(self.current_location.description)
        if self.current_location.enemy and self.current_location.enemy.is_alive():
            print(f"You encounter a {self.current_location.enemy.name}!")
            self.combat(self.current_location.enemy)
        while True:
            user_input = input("What do you want to do? ")
            if user_input.lower() == "quit":
                print("Thanks for playing!")
                return
            elif user_input.lower() == "move":
                self.move_to_new_location()
            elif user_input.lower().startswith("use "):
                item = user_input.split("use ", 1)[1]
                self.player.use_item(item)
            else:
                self.generate_response(user_input)
            self.generate_event()
            print(self.story)

    def move_to_new_location(self) -> None:
        print("Where would you like to go?")
        for i, location in enumerate(self.locations):
            print(f"{i}. {location.name}")
        choice = int(input("Enter the number of the location: "))
        self.current_location = self.locations[choice]
        print(f"You move to {self.current_location.name}.")
        print(self.current_location.description)
        if self.current_location.enemy and self.current_location.enemy.is_alive():
            print(f"You encounter a {self.current_location.enemy.name}!")
            self.combat(self.current_location.enemy)

    def combat(self, enemy: Enemy) -> None:
        while self.player.health > 0 and enemy.health > 0:
            print(f"You attack the {enemy.name}!")
            enemy.health -= self.player.attack
            if enemy.health <= 0:
                print(f"You've defeated the {enemy.name}!")
                self.player.gain_experience(10)
                return
            print(f"The {enemy.name} attacks you!")
            self.player.health -= enemy.attack
            if self.player.health <= 0:
                print("You've been defeated...")
                return
        print("The combat ends.")

    def generate_response(self, user_input: str) -> None:
        response = self.client.completion.create(
            engine="davinci",
            prompt=f"{self.story} {user_input}",
            temperature=0.7,
            max_tokens=100
        )
        self.story += response["choices"][0]["text"]

    def generate_event(self) -> None:
        if "interact" not in self.story.lower():
            event = random.choice(self.EVENTS)
            self.story += f"\n{event}"
