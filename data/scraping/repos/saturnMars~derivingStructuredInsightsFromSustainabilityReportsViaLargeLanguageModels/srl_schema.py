from kor import Object, Text, Number
from kor.nodes import Number, Object, Option, Selection, Text

from langchain import PromptTemplate
from os import path

# LOCAL IMPORTS
import sys
sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))
from _library.data_loader import load_esg_categories

    
labelled_sentences = [
    (
        "In accordance with our ambitious goal, the water withdrawal of the data center decreased remarkably from 3.874 million litres to 2.367 million litres across the past three years.",
        [
            {
                "esg_category": "Water", 
                "predicate": "Reduction of", 
                "object": "The water withdrawal of the data center by 1.507 million litres",
                "properties" : {
                    "sub_esg_category": "Water withdrawal",
                    "time": "The past three years",
                    "manner": "In accordance with our ambitious goal",
                }
            }
        ]
    ),( 
     "TotalEnergies introduced an innovative program at its European offices last year to address employees' concerns by creating a dedicated listening space.",
     [
          {
                "esg_category": "Employee Development", 
                "predicate": "Introduction of", 
                "object": "An innovative program",
                "properties" : {
                    "agent": "TotalEnergies",
                    "location": "The TotalEnergies' European offices",
                    "time": "Last year",
                    "manner": "By creating a dedicated listening space",
                    "purpose": "To address employees' concerns"
                }
          }
        ] 
    ),( 
     "In San Antonio, Texas, our company reduced significantly the potable water usage of the data center by around 20% throughout 2020, providing economic and environmental benefits.",
     [{
                "esg_category": "Water", 
                "predicate": "Reduction of", 
                "object": "The data center's potable water usage by around 20%",
                "properties" : {
                    "sub_esg_category": "Water usage", 
                    "agent": "Our company",
                    "location": "San Antonio, Texas",
                    "time": "2020",
                    "manner": "Providing economic and environmental benefits"
                }
            }
        ] 
    ), (
        "In 2019, the ethics training program was completed by over 95% of our employees with outstanding results at our American training centre.",
        [ 
         {
                "esg_category": "Employee Development", 
                "predicate": "Completion of", 
                "object": "The ethics training program",
                "properties" : {
                    "agent": "Over 95% of our employees",
                    "location": "American training centre",
                    "time": "2019",
                    "manner": "With outstanding result"
                }
            }
        ]
    ),(
        "Microsoft has invested €125 million in cutting-edge recycling technologies and smart waste management systems at its offices in Zwijndrecht, Belgium.",
        [{
                "esg_category": "Waste", 
                "predicate": "Investment in", 
                "object": "Cutting-edge recycling technologies and smart waste management systems",
                "properties" : {
                    "sub_esg_category": "Waste management", 
                    "agent": "Microsoft",
                    "location": "Microsoft's offices in Zwijndrecht, Belgium",
                    "manner": "By investing €125 million"
                }
            }
        ]
    )
]

schema = Object(
    id="esg_actions",
    description="actions related to corporate's environmental, social or governance aspects",
    attributes=[
        Selection(
            id="esg_category",
            description="an issue related to an ESG aspect",
            options=[Option(id = category_name.lower().replace(' ', '_').replace('-', '_'), description = category_name) 
                     for category_name in load_esg_categories()['rigobon_esg_taxonomy']]
        ),
        Text(
            id="predicate",
            description="a nominalized verb that affects the ESG-related category",
        ),
        Text(
            id="object",
            description="an entity related to the esg category that undergoes the predicate"
            #  A participant which the predicate characterizes as having something happen to it and as being affected by what happens to it",
        ),
        Object(
            id = "properties",
            description = 'characterizing properties of the topic, the predicate and the object',
            attributes = [
                Text(
                    id="sub_esg_category",
                    description = "a more specific issue related to an ESG aspect"
                ),
                Text(
                    id="agent",
                    description = "a participant that deliberately performs the action"
                ),
                Text(
                    id="location",
                    description = "where the action occurs"
                ),
                Text(
                    id="time",
                    description = "when the action occurs"
                ),
                Text(
                    id="manner",
                    description = "the way in which the action is carried out",
                ),
                Text(
                    id="purpose",
                    description = "the reason or goal for which the predicate is performed"
                )  
            ]
        )],
    many = True,
    examples = labelled_sentences
)

prompt_template = PromptTemplate(
    input_variables=["type_description", "format_instructions"],
    template=(
        "Your goal is to extract structured information from the user's input that matches the form described below. When extracting information please make sure it matches the type information exactly. Do not add any attributes that do not appear in the schema shown below.\n\n"
        "{type_description}\n\n"
        "{format_instructions}\n\n"
    )
)

prompt_template_ = PromptTemplate(
    input_variables=["type_description", "format_instructions"],
    template=(
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n"
        "Your goal is to extract structured semantic information from the user's input that matches the form described below. You might also exploit syntactical dependencies, but preserve semantical matches between the attribute-value pairs. Please disambiguate sentence boundaries and consider each sentence independent avoiding mixing semantic information. When extracting information please make sure it matches the type of information exactly. Do not add any attributes that do not appear in the schema shown below.\n\n"
        "{type_description}\n\n"
        "{format_instructions}\n\n### Input:\n"
    )
)