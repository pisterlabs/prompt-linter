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

prompt_template = """
You are part of the llm-based intent classification component used in a Rasa-based dialog engine.
Please use the format that Rasa uses for the intent and intent_ranking. Here is an example.
{{ "intent": {{"name": "test", "confidence": 1.0}}, intent_ranking: [{{"name": "test", "confidence": 1.0}}] }}

Please classify the following intents wrapped in brackets [] and create the intent object as well as the intent ranking object.
{0}

The items in the intent list can also include a description of the intent in the following format.
intent_name "intent_description"
You can use the description to help you classify the intent.

Please classify from the following text provided as user message and return the result as a JSON object as described in the example.

"""

@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER], is_trainable=False
)
class IntentClassifier(GraphComponent):
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
        else:
            self.intents = []

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

        self.prompt = prompt_template.format(self.intents)

    def train(self, training_data: TrainingData) -> Resource:
        pass

    def process_training_data(self, training_data: TrainingData) -> TrainingData:

        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        # This is the method which Rasa Open Source will call during inference.
        
        for message in messages:
            message = self.classify_intents_with_llm(message)

        return messages
    
    def classify_intents_with_llm(self, message: Message) -> Message:

        text = message.get("text")

        # send text to LLM model

        # The main disadvanteges of the LLM models are that the response time is slow.
        chat_completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": text}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # get intent and intent_ranking from LLM model

        llm_response_as_string = chat_completion.choices[0]['message']['content']

        llm_response = json.loads(llm_response_as_string)

        intent_dict = llm_response["intent"]

        message.set("intent", intent_dict, add_to_output=True)

        intent_ranking_dict = llm_response["intent_ranking"]

        message.set("intent_ranking", intent_ranking_dict, add_to_output=True)

        return message