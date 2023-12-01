from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate

class twn:

    def __init__(self, ui, chat, world):
        # Create UI
        self.ui = ui

        # Create Chat Model
        self.chat_llm = chat

        # Create World
        self.world = world

        # Generate Schema
        self.sprawl = ResponseSchema(name="sprawl", description="Sprawl Type (ex. Urban, Rural, etc)")
        self.name = ResponseSchema(name="name", description="Town Name. Be creative, and make the name sound fantasy by using prefixes from various languages")
        self.population = ResponseSchema(name="population", description="Number of citizens (25 to 100,000 inclusive, and try to make the number seem random by ensuring it is not rounded)")
        self.architecture = ResponseSchema(name="architecture", description="Architectural Style (ex. Gothic, Modern, Steampunk, etc)")
        self.industries = ResponseSchema(name="industries", description="Main Industries (ex. Fishing, Mining, Farming, etc)")
        self.lore = ResponseSchema(name="lore", description="Town Lore (1-3 Sentences)") 
        self.governing = ResponseSchema(name="governing", description="Describe the governing body and political state of the town, and perhaps name it (1-3 Sentences)")
        self.quests = ResponseSchema(name="quests", description="Describe the quests that the party may find in the town (1-3 Sentences)")
        self.climate = ResponseSchema(name="climate", description="Climate (ex. Temperate, Tropical, etc)")
        self.response_schema = [self.sprawl, self.name, self.population, self.architecture, self.industries, self.lore, self.governing, self.quests, self.climate]
        
        # Create Schema Parser
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schema)
        self.format_instructions = self.output_parser.get_format_instructions()

        # Create Template
        self.template_string = """You are an expert Dungeon Master for Dungeons and Dragons Fifth Edition \ 
        You come up with memorable and impactful towns/regions for your D&D campaign.
        
        Create a concept for a town or region that your party may stumble upon during their travels.

        When creating this town, be sure to include the following world information in your worldbuilding process.

        {world_info}   
        
        {format_instructions}"""


    def generate(self):

        # Create Prompt
        self.prompt = ChatPromptTemplate.from_template(template=self.template_string)

        # Create Messages
        self.messages = self.prompt.format_messages(format_instructions=self.format_instructions, world_info=self.world.loadWorld(self))

        # Parse Response
        self.response = self.chat_llm(self.messages)
        self.response_as_dict = self.output_parser.parse(self.response.content)
    
        # Update Labels
        self.ui.twnGovTypeLabel.setText(self.response_as_dict["governing"])
        self.ui.twnNameLabel.setText(self.response_as_dict["name"])
        self.ui.twnPopLabel.setText(self.response_as_dict["population"])
        self.ui.twnSprawlLabel.setText(self.response_as_dict["sprawl"])
        self.ui.twnArcLabel.setText(self.response_as_dict["architecture"])
        self.ui.twnClimLabel.setText(self.response_as_dict["climate"])
        self.ui.twnIndLabel.setText(self.response_as_dict["industries"])
        self.ui.twnLoreLabel.setText(self.response_as_dict["lore"])
        self.ui.twnQuestLabel.setText(self.response_as_dict["quests"])
        

    def save(self):

        # Create File
        self.filename = "saves/towns/" + self.response_as_dict["name"].lower().replace(" ", "_") + ".txt"
        self.file = open(self.filename, "w")

        # Write to File
        self.file.write("Name: " + self.response_as_dict["name"] + "\n")
        self.file.write("Population: " + self.response_as_dict["population"] + "\n")
        self.file.write("Sprawl: " + self.response_as_dict["sprawl"] + "\n")
        self.file.write("Architecture: " + self.response_as_dict["architecture"] + "\n")
        self.file.write("Climate: " + self.response_as_dict["climate"] + "\n")
        self.file.write("Industries: " + self.response_as_dict["industries"] + "\n")
        self.file.write("Lore: " + self.response_as_dict["lore"] + "\n")
        self.file.write("Governing: " + self.response_as_dict["governing"] + "\n")
        self.file.write("Quests: " + self.response_as_dict["quests"] + "\n")

        # Close File
        self.file.close()
