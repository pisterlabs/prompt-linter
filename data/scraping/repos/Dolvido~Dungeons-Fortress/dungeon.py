from langchain import HuggingFaceHub, LLMChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.prompts import PromptTemplate

from firebase_admin import firestore
from treasure import Treasure

import random

# The firestore client should be initialized in the bot.py file. The Dungeon class uses the instance previously created.
# You don't need to initialize it again here. Please remove: db = firestore.client()

class Dungeon:
    def __init__(self, player, db):
        self.player = player
        self.history = []
        self.db = db
        self.repo_id_llm = "tiiuae/falcon-7b-instruct"  # Moved to init method to avoid hard-coding in multiple places
        self.depth = 0
        self.threat_level = 1
        self.max_threat_level = 5
        self.threat_level_multiplier = 1.5
        self.escape_chance = 0.1
        self.room_type = "start"
        self.chat_history = ChatMessageHistory()
        self.memory = ConversationBufferMemory(memory_key="adventure_history")

    def delete_dungeon(self, db):
        print("delete_dungeon")
        # Get a reference to the Dungeon document and then call the delete() method.
        dungeon_ref = db.collection('dungeons').document(self.player.name)
        dungeon_ref.delete()


    def start(self, db):
        random_temperature = random.uniform(0.5, 0.7)

        dungeon_llm = HuggingFaceHub(repo_id=self.repo_id_llm,
                                    model_kwargs={
                                        "temperature": random_temperature,
                                        "max_new_tokens": 250
                                    })

        prompt_template = "{adventure_history} Paint a vivid picture of a {adventure_type} adventure set in an ancient and mysterious dungeon. What atmosphere and characteristics define this dungeon?"
        dungeon_start_prompt = PromptTemplate(template=prompt_template,
                                                input_variables=["adventure_history", "adventure_type"])

        llm_chain = LLMChain(prompt=dungeon_start_prompt, llm=dungeon_llm, memory=self.memory)

        try:
            response = llm_chain.predict(adventure_type="dungeoneering")
            response += "\nDo you /continue or /flee?"
        except Exception as e:
            response = f"I couldn't generate a response due to the following error: {str(e)}"

        return response

    def continue_adventure(self, db):
        # use db ref to update depth 
        self.depth += 1
        doc_ref = db.collection('dungeons').document(self.player.name)
        doc_ref.update({'depth': self.depth})

        damage_taken = 0
        response = ""
        print(self.player.name +' is continuing the dungeon adventure at depth ' + str(self.depth))

        self.print_threat_level()

        weights = {
            "combat": self.threat_level,
            "treasure": max(1, 10 - self.threat_level),
            "nothing": max(1, 5 - self.threat_level),
            "escape":max(1, 10 - self.threat_level)
        }

        encounter = random.choices(
            population=["combat", "treasure", "nothing", "escape"], 
            weights=[weights["combat"], weights["treasure"], weights["nothing"], weights["escape"]],
            k=1
        )[0]
        self.room_type = encounter

        if (encounter == "combat"):
            print('encountered a combat encounter: ', self.player.name)
            response += self.combat_operation(db)

        if encounter == "treasure":
            print('encountered a treasure room: ', self.player.name)
            response += self.treasure_operation(db)

        if (encounter == "nothing"):
            print('encountered an empty room: ', self.player.name)
            response += self.no_encounter_operation(db)
        if encounter == "escape":
            print('encountered an escape room: ', self.player.name)
            response += self.escape_room_operation(db)        

        self.update_threat_level(db)
        self.save_dungeon(db)
        return response

    def print_threat_level(self):
        print(f"Threat Level: {self.threat_level}")
        response = f"\nThreat Level: {self.threat_level}"

    # Implement the remaining needed methods as per the task and recorrecting where needed
    def escape_room_operation(self, db):
        # Generate random temperature variable for the language model chain
        random_temperature = random.uniform(0.5, 0.7)

        dungeon_llm = HuggingFaceHub(repo_id=self.repo_id_llm,
                                    model_kwargs={
                                        "temperature": random_temperature,
                                        "max_new_tokens": 100
                                    })

        generate_escape_room_prompt = "{adventure_history} Amidst the labyrinthine passages, the adventurer discovers a concealed door adorned with {properties}. This is no ordinary room—it's an escape chamber. Elaborate on its enigmatic features."
        llm_escape_room_prompt = PromptTemplate(template=generate_escape_room_prompt,
                                                input_variables=["adventure_history", "properties"])

        print("generating escape room")
        escape_room_chain = LLMChain(prompt=llm_escape_room_prompt, llm=dungeon_llm, memory=self.memory)
        escape_room_description = escape_room_chain.predict(properties="bathed in white light")

        response = "\nESCAPE ROOM\n"
        response += escape_room_description
        response += "\nDo you /continue or /escape?"

         # Player flees the dungeon without losing any treasure
        return response

    def combat_operation(self, db):
        # Generate random temperature variable for the language model chain
        random_temperature = random.uniform(0.5, 0.7)

        dungeon_llm = HuggingFaceHub(repo_id=self.repo_id_llm,
                                    model_kwargs={
                                        "temperature": random_temperature,
                                        "max_new_tokens": 100
                                    })
        
        enemy_attributes = {
            "type": random.choice(["goblin", "troll", "dragon", "skeleton", "zombie"]),
            "weapon": random.choice(["claws", "sword", "magic", "fangs", "axe"]),
            "appearance": random.choice(["horrifying", "grotesque", "terrifying", "ghastly", "hideous"]),
            "strength": random.choice(["immense strength", "magical powers", "swift agility", "overwhelming numbers", "deadly precision"]),
            "weakness": random.choice(["fear of light", "slow movements", "limited vision", "low intelligence", "magic susceptibility"])
        }

        enemy_assembled_string = f'A {enemy_attributes["appearance"]} {enemy_attributes["type"]} wielding a {enemy_attributes["weapon"]} with {enemy_attributes["strength"]}, but has a {enemy_attributes["weakness"]}'
        print("Enemy string: " + enemy_assembled_string)
        generate_enemy_prompt = "{adventure_history} Deep within the dungeon, where distant cries and howls reverberate, the adventurer faces an impending menace. Emerging from the gloom is a {enemy}. Detail its fearsome aspects."
        llm_enemy_prompt = PromptTemplate(template=generate_enemy_prompt, input_variables=["adventure_history", "enemy"])

        print("generating enemy")
        enemy_chain = LLMChain(prompt=llm_enemy_prompt, llm=dungeon_llm, memory=self.memory)
        enemy_description = enemy_chain.predict(enemy=enemy_assembled_string)

        combat_status, combat_message = self.player.handle_combat(self.threat_level, db)

        print("generating narrative")
        print("enemy description: " + enemy_description)
        if combat_status == "won":
            combat_narrative = self.get_victory_narrative(enemy_assembled_string)
        else:  # if the combat_status is "lost"
            combat_narrative = self.get_defeat_narrative(enemy_assembled_string)

        response = f"\nCOMBAT ENCOUNTER\n"
        response += f"Enemy: {enemy_assembled_string}\n"

        response += enemy_description
        response += combat_narrative
        response += "\n"+combat_message

        return response

    def get_victory_narrative(self, enemy_description):
        print("get_victory_narrative")
        random_temperature = random.uniform(0.5, 0.7)  # You can adjust the temperature as needed

        dungeon_llm = HuggingFaceHub(repo_id=self.repo_id_llm,
                                        model_kwargs={
                                            "temperature": random_temperature,
                                            "max_new_tokens": 100
                                        })

        victory_prompt = "{adventure_history} Armed with courage and unparalleled skill, the hero confronts the {enemy_description}. Narrate the awe-inspiring moment when the hero triumphs over the creature."
        llm_victory_prompt = PromptTemplate(template=victory_prompt, input_variables=["adventure_history", "enemy_description"])

        victory_chain = LLMChain(prompt=llm_victory_prompt, llm=dungeon_llm, memory=self.memory)
        combat_narrative = victory_chain.predict(enemy_description=enemy_description.format())
    
        return combat_narrative

    def get_defeat_narrative(self, enemy_description):
        print("get_defeat_narrative")
        random_temperature = random.uniform(0.5, 0.7)  # You can adjust the temperature as needed

        dungeon_llm = HuggingFaceHub(repo_id=self.repo_id_llm,
                                        model_kwargs={
                                            "temperature": random_temperature,
                                            "max_new_tokens": 100
                                        })
        
        defeat_prompt = "{adventure_history} Despite the hero's valiant efforts, the {enemy_description} proves to be too powerful. Describe the tragic moment the hero is defeated by the beast."
        llm_defeat_prompt = PromptTemplate(template=defeat_prompt, input_variables=["adventure_history", "enemy_description"])

        defeat_chain = LLMChain(prompt=llm_defeat_prompt, llm=dungeon_llm, memory=self.memory)
        combat_narrative = defeat_chain.predict(enemy_description=enemy_description.format())
    
        return combat_narrative


    def treasure_operation(self, db):
        """
        Handles the operation where the adventure enters a treasure room.
        """
        random_temperature = random.uniform(0.5, 0.7)

        dungeon_llm = HuggingFaceHub(
            repo_id=self.repo_id_llm,
            model_kwargs={
                "temperature": random_temperature,
                "max_new_tokens": 150
            }
        )

        treasure_type = random.choice(["jewel", "artifact", "scroll", "potion", "grimoire"])
        material = random.choice(["gold", "silver", "diamond", "ruby"])
        origin = random.choice(["dwarven", "elvish", "dragon hoard"])
            
        # Create a Treasure object
        discovered_treasure = Treasure(treasure_type, material, origin)
            
        # Add the treasure to the player's inventory
        self.player.add_to_inventory(discovered_treasure, db)
            
        # Add the treasure discovery to the scene description
        response = f"\nYou discovered a {discovered_treasure}!"

        
        generate_treasure_prompt = "{adventure_history} In the depths of the ancient and mystical dungeon, amidst the eerie silence punctuated by the echoes of distant roars and clanks, a hidden chamber reveals itself. Shrouded in mystery, it harbors a {treasure}."  # Changed {treasure_assembled_string} to {treasure}
        llm_treasure_prompt = PromptTemplate(template=generate_treasure_prompt,
                                            input_variables=["adventure_history", "treasure"])  # Changed "treasure_assembled_string" to "treasure"

        # Create language model chain and run against our prompt
        treasure_chain = LLMChain(prompt=llm_treasure_prompt, llm=dungeon_llm, memory=self.memory)
        generated_treasure = treasure_chain.predict(treasure=str(discovered_treasure))  # Changed {treasure_assembled_string={treasure_assembled_string}} to treasure=treasure_assembled_string

        # Add the treasure to the player's inventory and database
        #self.add_treasure_to_db(discovered_treasure, db)

        response = "\nTREASURE ROOM\n"
        response += f"\nYou discovered a {discovered_treasure}!"
        response += generated_treasure

        return response

    def no_encounter_operation(self, db):
        """
        Handles the operation where the adventure enters an empty room.
        """
        random_temperature = random.uniform(0.5, 0.7)

        dungeon_llm = HuggingFaceHub(repo_id=self.repo_id_llm,
                                    model_kwargs={
                                        "temperature": random_temperature,
                                        "max_new_tokens": 250
                                    })

        continue_dungeon_prompt = "{adventure_history} Describe an {quality} room in detail."
        llm_prompt = PromptTemplate(template=continue_dungeon_prompt, input_variables=["adventure_history", "quality"])

        llm_chain = LLMChain(prompt=llm_prompt, llm=dungeon_llm, memory=self.memory)

        try:
            generated_description = llm_chain.run(quality="empty")
            self.history.append(generated_description)

        except Exception as e:
            generated_description = f"I couldn't generate a description due to the following error: {str(e)}"

        response = "\nEMPTY ROOM\n" + generated_description
        return response
    
    def add_treasure_to_db(self, treasure, db):
        """
        Add the treasure to the database.
        """
        # Convert the Treasure object to a dictionary
        print(type(treasure))
        treasure_data = treasure.to_dict()
        print(type(treasure_data))

        # Here add the logic to store the treasure_data dictionary in the database
        # This will depend on the exact database and structure you are using
        # For example, if using Firebase Firestore:
        # Then, store the dictionary in the Firestore database
        try:
            # Replace this with the actual path where you want to store the treasure data
            db.collection('players').document(self.player.name).collection('treasures').add(treasure_data)
            return "Treasure added to the database!"
        except Exception as e:
            return f"An error occurred when adding treasure to database: {e}"    

    def update_threat_level(self, db):
        """
        Update threat level exponentially and cap the threat level to the maximum value.
        """
        print("update_threat_level")
        self.threat_level *= self.threat_level_multiplier
        print("calculated threat level: " + str(self.threat_level))
        print("self.depth" + str(self.depth))
        self.threat_level = min(self.threat_level, self.max_threat_level)
        dungeon_ref = db.collection('dungeons').document(self.player.name)
        if dungeon_ref.get().exists:
            dungeon_ref.update({'threat_level': self.threat_level})

    @staticmethod
    def load_dungeon(player, db):
        print("load_dungeon")
        dungeon_data = db.collection('dungeons').document(player.name).get()
        if dungeon_data.exists:
            data = dungeon_data.to_dict()
            d = Dungeon(player, db)
            
            d.repo_id_llm = data.get('repo_id_llm')
            d.depth = data.get('depth')
            d.threat_level = data.get('threat_level')
            d.room_type = data.get('room_type')
            d.max_threat_level = data.get('max_threat_level')
            d.threat_level_multiplier = data.get('threat_level_multiplier')
            
            return d
        else:
            return None

    def save_dungeon(self, db: firestore.Client):
        print("save_dungeon")
        
        doc_ref = db.collection('dungeons').document(self.player.name)
            
        # Check if document exists
        if doc_ref.get().exists:
            print(f"Document for player {self.player.name} already exists.")
            
        data = {
            'repo_id_llm': self.repo_id_llm,
            'depth' : self.depth,
            'threat_level' : self.threat_level,
            'room_type' : self.room_type,
            'max_threat_level' : self.max_threat_level,
            'threat_level_multiplier' : self.threat_level_multiplier
        }
            
        doc_ref.set(data)

    def to_dict(self):
        return {
            'depth': self.depth,
            'threat_level': self.threat_level,
            'room_type': self.room_type,
            'player': self.player.name if self.player else None  # Store only the player's name to avoid circular reference
        }

