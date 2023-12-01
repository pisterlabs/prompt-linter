from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
import random

class enc:

    def __init__(self, ui, chat, world):
        # Create UI
        self.ui = ui

        # Create WOrld
        self.world = world

        # Create Chat Model
        self.chat_llm = chat

        # Generate Schema
        self.name = ResponseSchema(name="name", description="Encounter Name, capture tones of excitement and treachery (1-5 words only)")
        self.description = ResponseSchema(name="description", description="Description of how the encounter begins, and some objectives for the encounter (1-3 Sentences)")
        self.creatures = ResponseSchema(name="creatures", description="A comma separated list of beasts from the 5e Monster Manual, based on the Challenge Ratings specified.")
        self.response_schema = [self.name, self.description, self.creatures]

        # Create Schema Parser
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schema)
        self.format_instructions = self.output_parser.get_format_instructions()

        # Create Template
        self.template_string = """You are an expert Dungeon Master for Dungeons and Dragons Fifth Edition \ 
        You come up with engaging and exciting combat encounters that your party may engage with.
        
        Create an encounter that your party may stumble upon during their travels.

        When creating this encounter, be sure to include the following world information in your worldbuilding process.

        {challenge_rating}

        {world_info}   
        
        {format_instructions}"""

    def crCombos(self):
        challengeRating = self.ui.encCRSlider.value()
        creatures = self.ui.encNumCreatures.value()

        values = []
        for i in range(1, 31): values.append(i)

        failedGeneration = False
        res = []
        while challengeRating > 0:
            if creatures==1:
                res.append(challengeRating)
                creatures -= 1
                break
            counter = 0
            maxVal = challengeRating - (creatures-1)
            if maxVal < 1: 
                if values[0] == 1:
                    values.insert(0, 1/2)
                    values.insert(0, 1/4)
                    values.insert(0, 1/8)
                maxVal = challengeRating - (1/8)*(creatures-1)
            for val in values: 
                if val<=maxVal: counter += 1
            
            if counter==0 and challengeRating>0:
                failedGeneration = True
                break
            val = values[int(random.random()*counter)]
            if val < 1: val = values[counter-1];
            res.append(val)
            challengeRating -= val
            creatures -= 1
        if failedGeneration:
            return "CR Generation Failed, please enter ERROR into the encounter creatures list"
        return res

    def generate(self):

        # Create Prompt
        self.prompt = ChatPromptTemplate.from_template(template=self.template_string)

        self.crList = self.crCombos()
        self.crPrompt = "These are the following challenge rating of the monsters to generate: " + str(self.crList) + ". The encounter must have a total of " + str(self.ui.encNumCreatures.value()) + "creatures with a challenge rating of " + str(self.ui.encCRSlider.value())

        # Create Messages
        self.messages = self.prompt.format_messages(format_instructions=self.format_instructions, challenge_rating = self.crPrompt, world_info=self.world.loadWorld(self))
        # Parse Response
        self.response = self.chat_llm(self.messages)
        self.response_as_dict = self.output_parser.parse(self.response.content)

        # Update Labels
        self.ui.encNameLabel.setText(self.response_as_dict["name"])
        self.ui.encCreaturesLabel.setText(str(self.response_as_dict["creatures"]))
        self.ui.encDesLabel.setText(self.response_as_dict["description"])


    def save(self):

        # Create File
        self.filename = "saves/encounters/" + self.response_as_dict["name"].lower().replace(" ", "_") + ".txt"
        self.file = open(self.filename, "w")

        # Write to File
        self.file.write("Encounter: " + self.response_as_dict["name"] + "\n")
        self.file.write("Creatures: " + self.response_as_dict["creatures"] + "\n")
        self.file.write("Description: " + self.response_as_dict["description"] + "\n")

        # Close File
        self.file.close()


        

