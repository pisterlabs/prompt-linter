from typing import Dict, Text, Any, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

import os
import json
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Set this if you want to use another LLM API that replicates
# the OpenAI API specification like https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-api
#openai.api_base = ""

prompt_template = """
You are part of the llm-based intent classification and entity extraction component used in a Rasa-based dialog engine.

Please use the format that Rasa uses for the intent and intent_ranking. Here is an example.
{{ "intent": {{"name": "example_intent_name", "confidence": 1.0}}, intent_ranking: [{{"name": "example_intent_name", "confidence": 1.0}}] }}

Please classify the following intents wrapped in brackets [] and create the intent object as well as the intent ranking object.
{0}
The items in the intent list can also include a description of the intent in the following format.
intent_name "intent_description"
You can use the description to help you classify the intent.
The description may also include a list of entities that are typically associated with the intent.
The description of the intent may also include if entities should be extracted from the user message or not.

Please use the format that Rasa uses for entities
{{"entity": "entity_name", "value": "entity_value", "start": 0, "end": 4, "extractor": "DualIntentAndEntityLLM"}}.
The items in the entity list can also include a description of the entity in the following format.
entity_name "entity_description"
You can use the description to help you extract the entity.
Return the entities as a JSON array.

You should extract the following entities wrapped in brackets [].
{1}

Please classify the intents and extract the entities from the following text provided as user message.
Please return the intent object, the intent ranking object, and the entities as a JSON object of the following format.

{{
  "intent": {{
    "name": "example_intent_name",
    "confidence": 1.0
  }},
  "entities": [],
  "intent_ranking": [
    {{
      "name": "",
      "confidence": 1.0
    }}
  ]
}}

"""

@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR,
        DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER
    ],
    is_trainable=False
)
class DualIntentAndEntityLLM(GraphComponent):
    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:

        return cls(config)
    
    def __init__(self, config: Dict[Text, Any]):

        super().__init__()

        if "intents" in config:
            self.intents = config["intents"]
            print(self.intents)
        else:
            self.intents = []

        if "entities" in config:
            self.entities = config["entities"]
        else:
            self.entities = []

        if "model" in config:
            self.model = config["model"]
        else:
            self.model = "gpt-3.5-turbo"

        if "temperature" in config:
            self.temperature = config["temperature"]
        else:
            self.temperature = 0
        
        if "max_tokens" in config:
            self.max_tokens = config["max_tokens"]
        else:
            self.max_tokens = 1024

        self.prompt = prompt_template.format(self.intents, self.entities)


    def train(self, training_data: TrainingData) -> Resource:
        pass

    def process_training_data(self, training_data: TrainingData) -> TrainingData:

        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        # This is the method which Rasa Open Source will call during inference.
        
        for message in messages:
            message = self.extract_entities_with_llm(message)

        return messages
    
    def extract_entities_with_llm(self, message: Message) -> Message:

        text = message.get("text")

        # send text to LLM model

        # The main disadvantage of the LLM models are that the response time is often slow.
        chat_completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": text}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # classify intents and extract entities using an LLM model

        llm_response_as_string = chat_completion.choices[0]['message']['content']

        llm_response = json.loads(llm_response_as_string)

        intent_dict = llm_response["intent"]

        message.set("intent", intent_dict, add_to_output=True)

        intent_ranking_dict = llm_response["intent_ranking"]

        message.set("intent_ranking", intent_ranking_dict, add_to_output=True)

        entities = llm_response["entities"]

        message.set("entities", entities, add_to_output=True)

        return message