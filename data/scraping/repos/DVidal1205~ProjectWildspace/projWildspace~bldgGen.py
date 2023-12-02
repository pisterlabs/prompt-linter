from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate

class bldg:

    def __init__(self, ui, chat, world):
        # Create UI
        self.ui = ui

        # Create World
        self.world = world

        # Create Chat Model
        self.chat_llm = chat

        # Generate Schema
        self.type = ResponseSchema(name="type", description="Building Type (ex. Tavern, Blacksmith, Fisher, Market Vendor, etc)")
        self.name = ResponseSchema(name="name", description="Building Name. Be creative, and take inspiration from building type")
        self.architecture = ResponseSchema(name="architecture", description="Architectural Style (ex. Gothic, Modern, Steampunk, etc)")
        self.ambience = ResponseSchema(name="ambience", description="Description of the interior theme, style, and decor (5-15 words)")
        self.size = ResponseSchema(name="size", description="Size of the building (ex. Small, Medium, Large, etc)")
        self.traffic = ResponseSchema(name="traffic", description="Amount of foot traffic the building receives (ex. Low, Medium, High, etc)")
        self.description = ResponseSchema(name="description", description="Description of what is going on in the building, and what the player sees when they enter (1-3 Sentences)")
        self.vendor = ResponseSchema(name="vendor", description="Building Vendor Description. Include their name, race, and a brief description of their personality/appearance. (1-2 Sentences)")
        self.goods = ResponseSchema(name="goods", description="Description of the goods sold in the building. (1-2 Sentences)")
        self.response_schema = [self.type, self.name, self.architecture, self.ambience, self.size, self.traffic, self.description, self.vendor, self.goods]

        # Create Schema Parser
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schema)
        self.format_instructions = self.output_parser.get_format_instructions()

        # Create Template
        self.template_string = """You are an expert Dungeon Master for Dungeons and Dragons Fifth Edition \ 
        You come up with catchy and memorable scenes and buildings for your D&D campaign.
        
        Create a concept for a place your party may stumble upon during their travels.

        For naming schemes, use prefixes from various languages to create names, or create combinations of words that sound cool together.

        When making this building/scence, be sure to contextualize the following information about the world as best as possible, i.e, include the world into your generation of the building.

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
        self.ui.bldgNameLabel.setText(self.response_as_dict["name"])
        self.ui.bldgTypeLabel.setText(self.response_as_dict["type"])
        self.ui.bldgVendorLabel.setText(self.response_as_dict["vendor"])
        self.ui.bldgArcLabel.setText(self.response_as_dict["architecture"])
        self.ui.bldgAmbLabel.setText(self.response_as_dict["ambience"])
        self.ui.bldgSizeLabel.setText(self.response_as_dict["size"])
        self.ui.bldgTrafficLabel.setText(self.response_as_dict["traffic"])
        self.ui.bldgDesLabel.setText(self.response_as_dict["description"])
        self.ui.bldgGoodsLabel.setText(self.response_as_dict["goods"])

    def save(self):

        # Create File
        self.filename = "saves/buildings/" + self.response_as_dict["name"].lower().replace(" ", "_") + ".txt"
        self.file = open(self.filename, "w")

        # Write to File
        self.file.write("Name: " + self.response_as_dict["name"] + "\n")
        self.file.write("Type: " + self.response_as_dict["type"] + "\n")
        self.file.write("Ambience: " + self.response_as_dict["ambience"] + "\n")
        self.file.write("Architecture: " + self.response_as_dict["architecture"] + "\n")
        self.file.write("Size: " + self.response_as_dict["size"] + "\n")
        self.file.write("Traffic: " + self.response_as_dict["traffic"] + "\n")
        self.file.write("Vendor: " + self.response_as_dict["vendor"] + "\n")
        self.file.write("Description: " + self.response_as_dict["description"] + "\n")
        self.file.write("Goods: " + self.response_as_dict["goods"] + "\n")

        # Close File
        self.file.close()


        
