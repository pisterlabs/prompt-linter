from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
import random
import math

class npc:

    def __init__(self, ui, chat, world):
        # Create UI
        self.ui = ui

        # Create World
        self.world = world

        # Create Chat Model
        self.chat_llm = chat

        self.race_info = [
                {"Race": "Aarakockra", "Low Age": 3, "High Age": 30, "Low Height": 60, "High Height": 72},
                {"Race": "Aasimar", "Low Age": 18, "High Age": 100, "Low Height": 60, "High Height": 72},
                {"Race": "Bugbear", "Low Age": 16, "High Age": 80, "Low Height": 72, "High Height": 96},
                {"Race": "Centaur", "Low Age": 15, "High Age": 100, "Low Height": 72, "High Height": 84},
                {"Race": "Changeling", "Low Age": 20, "High Age": 80, "Low Height": 60, "High Height": 72},
                {"Race": "Dragonborn", "Low Age": 15, "High Age": 80, "Low Height": 60, "High Height": 84},
                {"Race": "Chromatic Dragonborn", "Low Age": 15, "High Age": 80, "Low Height": 60, "High Height": 84},
                {"Race": "Gem Dragonborn", "Low Age": 15, "High Age": 80, "Low Height": 60, "High Height": 84},
                {"Race": "Metallic Dragonborn", "Low Age": 15, "High Age": 80, "Low Height": 60, "High Height": 84},
                {"Race": "Dwarves", "Low Age": 50, "High Age": 350, "Low Height": 36, "High Height": 48},
                {"Race": "Duergar", "Low Age": 50, "High Age": 350, "Low Height": 36, "High Height": 48},
                {"Race": "Hill Dwarf", "Low Age": 50, "High Age": 350, "Low Height": 36, "High Height": 48},
                {"Race": "Mountain Dwarf", "Low Age": 50, "High Age": 350, "Low Height": 36, "High Height": 48},
                {"Race": "Elves", "Low Age": 100, "High Age": 750, "Low Height": 60, "High Height": 72},
                {"Race": "Astral Elf", "Low Age": 100, "High Age": 750, "Low Height": 60, "High Height": 72},
                {"Race": "Drow (Dark Elf)", "Low Age": 100, "High Age": 750, "Low Height": 60, "High Height": 72},
                {"Race": "Eladrin", "Low Age": 100, "High Age": 750, "Low Height": 60, "High Height": 72},
                {"Race": "High Elf", "Low Age": 100, "High Age": 750, "Low Height": 60, "High Height": 72},
                {"Race": "Sea Elf", "Low Age": 100, "High Age": 750, "Low Height": 60, "High Height": 72},
                {"Race": "Shadar-Kai", "Low Age": 100, "High Age": 750, "Low Height": 60, "High Height": 72},
                {"Race": "Wood Elf", "Low Age": 100, "High Age": 750, "Low Height": 60, "High Height": 72},
                {"Race": "Fairy", "Low Age": 20, "High Age": 200, "Low Height": 48, "High Height": 72},
                {"Race": "Firbolg", "Low Age": 30, "High Age": 500, "Low Height": 72, "High Height": 96},
                {"Race": "Genasi", "Low Age": 18, "High Age": 100, "Low Height": 60, "High Height": 72},
                {"Race": "Air Genasi", "Low Age": 18, "High Age": 100, "Low Height": 60, "High Height": 72},
                {"Race": "Earth Genasi", "Low Age": 18, "High Age": 100, "Low Height": 60, "High Height": 72},
                {"Race": "Fire Genasi", "Low Age": 18, "High Age": 100, "Low Height": 60, "High Height": 72},
                {"Race": "Water Genasi", "Low Age": 18, "High Age": 100, "Low Height": 60, "High Height": 72},
                {"Race": "Giff", "Low Age": 18, "High Age": 80, "Low Height": 60, "High Height": 84},
                {"Race": "Gith", "Low Age": 18, "High Age": 80, "Low Height": 60, "High Height": 84},
                {"Race": "Githyanki", "Low Age": 18, "High Age": 80, "Low Height": 60, "High Height": 84},
                {"Race": "Githzerai", "Low Age": 18, "High Age": 80, "Low Height": 60, "High Height": 84},
                {"Race": "Gnomes", "Low Age": 20, "High Age": 500, "Low Height": 36, "High Height": 48},
                {"Race": "Autognome", "Low Age": 20, "High Age": 500, "Low Height": 36, "High Height": 48},
                {"Race": "Deep Gnome (Svirfneblin)", "Low Age": 20, "High Age": 500, "Low Height": 36, "High Height": 48},
                {"Race": "Forest Gnome", "Low Age": 20, "High Age": 500, "Low Height": 36, "High Height": 48},
                {"Race": "Rock Gnome", "Low Age": 20, "High Age": 500, "Low Height": 36, "High Height": 48},
                {"Race": "Goblin", "Low Age": 10, "High Age": 60, "Low Height": 36, "High Height": 48},
                {"Race": "Goliath", "Low Age": 18, "High Age": 100, "Low Height": 72, "High Height": 96},
                {"Race": "Grung", "Low Age": 18, "High Age": 80, "Low Height": 24, "High Height": 48},
                {"Race": "Hadozee", "Low Age": 15, "High Age": 60, "Low Height": 60, "High Height": 84},
                {"Race": "Half-Elf", "Low Age": 20, "High Age": 180, "Low Height": 60, "High Height": 72},
                {"Race": "Half-Orc", "Low Age": 14, "High Age": 75, "Low Height": 60, "High Height": 84},
                {"Race": "Halflings", "Low Age": 20, "High Age": 250, "Low Height": 24, "High Height": 36},
                {"Race": "Ghostwise Halfling", "Low Age": 20, "High Age": 250, "Low Height": 24, "High Height": 36},
                {"Race": "Lightfoot Halfling", "Low Age": 20, "High Age": 250, "Low Height": 24, "High Height": 36},
                {"Race": "Stout Halfling", "Low Age": 20, "High Age": 250, "Low Height": 24, "High Height": 36},
                {"Race": "Harengon", "Low Age": 18, "High Age": 100, "Low Height": 60, "High Height": 72},
                {"Race": "Hobgoblin", "Low Age": 20, "High Age": 80, "Low Height": 60, "High Height": 72},
                {"Race": "Human", "Low Age": 18, "High Age": 100, "Low Height": 48, "High Height": 84},
                {"Race": "Kalashtar", "Low Age": 20, "High Age": 100, "Low Height": 60, "High Height": 72},
                {"Race": "Kender", "Low Age": 20, "High Age": 100, "Low Height": 36, "High Height": 48},
                {"Race": "Kenku", "Low Age": 12, "High Age": 60, "Low Height": 48, "High Height": 60},
                {"Race": "Kobold", "Low Age": 6, "High Age": 120, "Low Height": 24, "High Height": 36},
                {"Race": "Leonin", "Low Age": 18, "High Age": 100, "Low Height": 72, "High Height": 84},
                {"Race": "Lizardfolk", "Low Age": 14, "High Age": 60, "Low Height": 60, "High Height": 84},
                {"Race": "Locathah", "Low Age": 10, "High Age": 80, "Low Height": 48, "High Height": 72},
                {"Race": "Loxodon", "Low Age": 50, "High Age": 450, "Low Height": 72, "High Height": 96},
                {"Race": "Minotaur", "Low Age": 20, "High Age": 150, "Low Height": 72, "High Height": 84},
                {"Race": "Orc", "Low Age": 12, "High Age": 80, "Low Height": 60, "High Height": 84},
                {"Race": "Owlin", "Low Age": 15, "High Age": 80, "Low Height": 60, "High Height": 72},
                {"Race": "Plasmoids", "Low Age": 10, "High Age": 100, "Low Height": 48, "High Height": 72},
                {"Race": "Satyr", "Low Age": 18, "High Age": 100, "Low Height": 60, "High Height": 72},
                {"Race": "Shifter", "Low Age": 18, "High Age": 100, "Low Height": 60, "High Height": 72},
                {"Race": "Simic Hybrid", "Low Age": 18, "High Age": 100, "Low Height": 60, "High Height": 72},
                {"Race": "Tabaxi", "Low Age": 18, "High Age": 100, "Low Height": 60, "High Height": 72},
                {"Race": "Thri-Kreen", "Low Age": 5, "High Age": 30, "Low Height": 72, "High Height": 84},
                {"Race": "Tiefling", "Low Age": 18, "High Age": 100, "Low Height": 60, "High Height": 72},
                {"Race": "Tortle", "Low Age": 15, "High Age": 60, "Low Height": 48, "High Height": 60},
                {"Race": "Triton", "Low Age": 18, "High Age": 200, "Low Height": 60, "High Height": 72},
                {"Race": "Vedalken", "Low Age": 20, "High Age": 100, "Low Height": 60, "High Height": 72},
                {"Race": "Verdan", "Low Age": 18, "High Age": 100, "Low Height": 60, "High Height": 72},
                {"Race": "Warforged", "Low Age": 2, "High Age": 30, "Low Height": 60, "High Height": 84},
                {"Race": "Yuan-Ti Pureblood", "Low Age": 18, "High Age": 100, "Low Height": 60, "High Height": 72},
        ]
        self.hairColors = ["Black", "Brown", "Dark Brown", "Light Brown", "Golden Brown", "Chestnut Brown",
        "Auburn", "Red", "Strawberry Blonde", "Blonde", "Platinum Blonde", "Dirty Blonde",
        "Ash Blonde", "Honey Blonde", "Sandy Blonde", "Silver", "Gray", "Salt and Pepper",
        "White", "Blue", "Navy Blue", "Teal", "Green", "Emerald Green", "Forest Green",
        "Olive Green", "Mint Green", "Pink", "Hot Pink", "Bubblegum Pink", "Lavender",
        "Purple", "Violet", "Indigo", "Lilac", "Orchid", "Orchid Pink", "Peach", "Copper",
        "Mahogany", "Burgundy", "Maroon", "Wine", "Rust", "Cinnamon", "Amber", "Honey",
        "Tawny", "Platinum", "Lavender Gray", "Steel Gray", "Charcoal", "Rainbow", "Ombre",
        "Two-Tone", "Highlights", "Lowlights", "Balayage", "Unicorn", "Mermaid", "Neon",
        "Pastel", "Iridescent", "Opal", "Silver Fox", "Natural"]
        self.eyeColors = [
        "Amber", "Blue", "Brown", "Green", "Gray", "Hazel", "Black", "Red", "Violet",
        "Aqua", "Teal", "Turquoise", "Gold", "Silver", "Copper", "Topaz", "Emerald",
        "Sapphire", "Ruby", "Opal", "Onyx", "Peridot", "Aquamarine", "Jade", "Bronze",
        "Chestnut", "Olive", "Lavender", "Indigo", "Honey", "Pink", "Platinum", "White",
        "Mixed", "Multicolored", "Other"]
        self.builds = [
        "Muscular", "Skinny", "Cunning", "Bulky", "Lean", "Athletic",
        "Tall", "Short", "Average", "Stocky", "Chubby", "Slender", "Hulking",
        "Compact", "Stout", "Lithe", "Robust", "Gangly", "Toned", "Gnarled"]
        self.genderList = ["Male", "Female", "Non-Binary", "Construct"]
        # Generate Class and Subclass
        self.classesAndSubclasses = [
        ["Artificer", ["Alchemist", "Artillerist", "Battle Smith"]],
        ["Barbarian", ["Path of the Ancestral Guardian", "Path of the Battlerager", "Path of the Beast",
                    "Path of the Berserker", "Path of the Storm Herald", "Path of the Totem Warrior",
                    "Path of the Wild Soul"]],
        ["Bard", ["College of Creation", "College of Eloquence", "College of Glamour", "College of Lore",
                "College of Swords", "College of Valor", "College of Whispers"]],
        ["Cleric", ["Arcana Domain", "Death Domain", "Forge Domain", "Grave Domain", "Knowledge Domain",
                    "Life Domain", "Light Domain", "Nature Domain", "Order Domain", "Peace Domain",
                    "Tempest Domain", "Trickery Domain", "Twilight Domain", "War Domain"]],
        ["Druid", ["Circle of Dreams", "Circle of Spores", "Circle of Stars", "Circle of the Shepherd",
                "Circle of the Land", "Circle of the Moon"]],
        ["Fighter", ["Arcane Archer", "Banneret", "Battle Master", "Cavalier", "Champion", "Echo Knight",
                    "Eldritch Knight", "Psi Knight", "Rune Knight", "Samurai"]],
        ["Monk", ["Way of the Astral Self", "Way of the Drunken Master", "Way of Mercy", "Way of the Open Hand",
                "Way of the Shadow", "Way of the Sun Soul", "Way of the Four Elements", "Way of the Kensei"]],
        ["Paladin", ["Oath of the Ancients", "Oath of Conquest", "Oath of the Crown", "Oath of Devotion",
                    "Oath of Redemption", "Oath of Vengeance"]],
        ["Ranger", ["Beast Master", "Fey Wanderer", "Gloom Stalker", "Horizon Walker", "Hunter", "Monster Slayer",
                    "Swarmkeeper"]],
        ["Rogue", ["Arcane Trickster", "Assassin", "Inquisitive", "Mastermind", "Phantom", "Scout", "Soulknife",
                "Swashbuckler", "Thief"]],
        ["Sorcerer", ["Aberrant Mind", "Clockwork Soul", "Divine Soul", "Draconic Bloodline", "Shadow Magic",
                    "Storm Sorcery", "Wild Magic"]],
        ["Warlock", ["The Archfey", "The Celestial", "The Fathomless", "The Fiend", "The Genie", "The Great Old One",
                    "The Hexblade", "The Undying"]],
        ["Wizard", ["Bladesinging", "Chronurgy Magic", "Graviturgy Magic", "School of Abjuration", "School of Conjuration",
                    "School of Divination", "School of Enchantment", "School of Evocation", "School of Illusion",
                    "School of Necromancy", "School of Transmutation", "War Magic"]]]
        self.alignmentList = [
            "Chaotic Neutral",
            "Chaotic Evil", 
            "Chaotic Good",
            "True Neutral",
            "Neutral Good",
            "Neutral Evil",
            "Lawful Good",
            "Lawful Neutral",
            "Lawful Evil"
        ]     


        # Generate Schema
        self.first_name_schema = ResponseSchema(name="first_name", description="Character first name")
        self.last_name_schema = ResponseSchema(name="last_name", description="Character last name")
        self.background_schema = ResponseSchema(name="background", description="Character Lore/Background (1-3 sentences))")
        self.motivation_schema = ResponseSchema(name="motivation", description="Character Motivation (1-3 sentences)")
        self.quirk_schema = ResponseSchema(name="quirk", description="Character Roleplay Quirk (1-3 sentences)")
        self.fashion_schema = ResponseSchema(name="fashion", description="Fashion Style (1-3 sentences)")        
        self.response_schema = [self.first_name_schema, self.last_name_schema, self.background_schema, self.motivation_schema, self.quirk_schema, self.fashion_schema]

        # Create Schema Parser
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schema)
        self.format_instructions = self.output_parser.get_format_instructions()

        # Create Template
        self.template_string = """You are an expert Dungeon Master for Dungeons and Dragons Fifth Edition \ 
        You come up with catchy and memorable ideas for a Dungeons and Dragons Campaign.
        
        Create a character concept for an NPC your party may encounter using the following information.

        For naming schemes, use prefixes from various languages to create names.

        When making this character, be sure to contextualize the following information about the world as best as possible, i.e, include the world into your generation of the character.

        {character_info}

        {world_info}   
        
        {format_instructions}"""

    def generate(self):

        # Create Empty Dict
        self.character_traits = {}

        # Create Prompt
        self.prompt = ChatPromptTemplate.from_template(template=self.template_string)

        # Race
        self.racial = random.choice(self.race_info)
        self.character_traits["race"] = self.racial["Race"]

        # Class and Subclass
        self.charClassList = random.choice(self.classesAndSubclasses)
        self.charClass = self.charClassList[0]
        self.subClass = random.choice(self.charClassList[1])
        self.character_traits["class"] = self.charClass
        self.character_traits["subclass"] = self.subClass
        
        # Alignment
        self.alignment = random.choice(self.alignmentList)
        self.character_traits["alignment"] = self.alignment

        # Age
        self.lowAge = int(self.racial["Low Age"])
        self.highAge = int(self.racial["High Age"])
        self.age = str(random.randint(self.lowAge, self.highAge))
        self.character_traits["age"] = self.age

        # Height
        self.lowHeight = int(self.racial["Low Height"])
        self.highHeight = int(self.racial["High Height"])
        self.height = random.randint(self.lowHeight, self.highHeight)
        self.heightFt = math.floor(self.height/12)
        self.heightIn = self.height % 12
        self.finalHeight = str(self.heightFt) + "'" + str(self.heightIn) + "\""
        self.character_traits["height"] = self.finalHeight

        # Hair
        self.hair = random.choice(self.hairColors)
        self.character_traits["hair"] = self.hair

        # Eyes
        self.eyes = random.choice(self.eyeColors)
        self.character_traits["eyes"] = self.eyes

        # Build
        self.build = random.choice(self.builds)
        self.character_traits["build"] = self.build

        # Gender
        self.gender = random.choice(self.genderList)
        self.character_traits["gender"] = self.gender

        # Create Messages
        self.messages = self.prompt.format_messages(format_instructions=self.format_instructions, character_info=self.character_traits, world_info=self.world.loadWorld(self))
        # Parse Response
        self.response = self.chat_llm(self.messages)
        self.response_as_dict = self.output_parser.parse(self.response.content)

        # Add to character traits
        self.character_traits["first_name"] = self.response_as_dict["first_name"]
        self.character_traits["last_name"] = self.response_as_dict["last_name"]
        self.character_traits["background"] = self.response_as_dict["background"]
        self.character_traits["motivation"] = self.response_as_dict["motivation"]
        self.character_traits["quirk"] = self.response_as_dict["quirk"]
        self.character_traits["fashion"] = self.response_as_dict["fashion"]

        # Update Labels
        self.ui.npcFirstNameLabel.setText(self.character_traits["first_name"])
        self.ui.npcLastNameLabel.setText(self.character_traits["last_name"])
        self.ui.genderLabel.setText(self.character_traits["gender"])
        self.ui.npcClassLabel.setText(self.character_traits["class"])
        self.ui.npcSubclassLabel.setText(self.character_traits["subclass"])
        self.ui.npcAlignmentLabel.setText(self.character_traits["alignment"])
        self.ui.npcRraceLabel.setText(self.character_traits["race"])
        self.ui.npcHeightLabel.setText(self.character_traits["height"])
        self.ui.npcAgeLabel.setText(self.character_traits["age"])
        self.ui.npcEyesLabel.setText(self.character_traits["eyes"])
        self.ui.npcBuildLabel.setText(self.character_traits["build"])
        self.ui.npcHairLabel.setText(self.character_traits["hair"])
        self.ui.npcBackgroundLabel.setText(self.character_traits["background"])
        self.ui.npcGoalsLabel.setText(self.character_traits["motivation"])
        self.ui.npcQuirksLabel.setText(self.character_traits["quirk"])
        self.ui.npcFashionLabel.setText(self.character_traits["fashion"])

                
    def save(self):

        # Create File
        self.filename = "saves/npcs/" + self.character_traits["first_name"].lower() + "_" + self.character_traits["last_name"].lower() + ".txt"
        self.file = open(self.filename, "w")

        # Character Traints
        self.file.write("Character Traits \n")
        self.file.write("First Name: " + self.character_traits["first_name"] + "\n")
        self.file.write("Last Name: " + self.character_traits["last_name"] + "\n")
        self.file.write("Gender: " + self.character_traits["gender"] + "\n")
        self.file.write("Alignment: " + self.character_traits["alignment"] + "\n")
        self.file.write("Class: " + self.character_traits["class"] + "\n")
        self.file.write("Subclass: " + self.character_traits["subclass"] + "\n\n")

        # Physical Traits
        self.file.write("Physical Traits \n")
        self.file.write("Race: " + self.character_traits["race"] + "\n")
        self.file.write("Height: " + self.character_traits["height"] + "\n")
        self.file.write("Age: " + self.character_traits["age"] + "\n")
        self.file.write("Build: " + self.character_traits["build"] + "\n")
        self.file.write("Hair Color: " + self.character_traits["hair"] + "\n")
        self.file.write("Eye Color: " + self.character_traits["eyes"] + "\n\n")
        
        # Character Traits
        self.file.write("Character Traits \n")
        self.file.write("Background: " + self.character_traits["background"] + "\n")
        self.file.write("Motivation: " + self.character_traits["motivation"] + "\n")
        self.file.write("Quirk: " + self.character_traits["quirk"] + "\n")
        self.file.write("Fashion: " + self.character_traits["fashion"] + "\n\n")

        # Close File
        self.file.close()

        
