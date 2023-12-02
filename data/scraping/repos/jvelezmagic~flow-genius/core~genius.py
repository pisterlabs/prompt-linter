import json

from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import messages_to_dict
from pydantic import BaseModel, validator

from app.config.settings import settings


class ActionParameter(BaseModel):
    field: str
    format: str
    required: bool


class Intent(BaseModel):
    name: str
    description: str
    action_url: str
    action_method: str
    action_parameters: list[ActionParameter]


class FlowGenius(BaseModel):
    conversation_id: str
    intents: list[Intent] = None
    history: RedisChatMessageHistory = None
    memory_key: str = "history"
    memory: ConversationBufferMemory = None
    conversation_llm: ChatOpenAI = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    intent_llm: ChatOpenAI = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    schema_llm: ChatOpenAI = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    confirmation_llm: ChatOpenAI = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    verbose: bool = settings.verbose

    class Config:
        arbitrary_types_allowed = True

    @validator("intents", pre=True, always=True)
    def set_intents(cls, v, values):
        if v is not None:
            assert isinstance(v, list[Intent])
            return v

        intents = cls.load_intents_from_json(path=settings.intents_path)
        return intents

    @validator("history", pre=True, always=True)
    def set_history(cls, v, values):
        if v is not None:
            assert isinstance(v, RedisChatMessageHistory)
            return v

        history = RedisChatMessageHistory(
            session_id=values["conversation_id"], url=settings.redis_url
        )

        return history

    @validator("memory", pre=True, always=True)
    def set_memory(cls, v, values):
        if v is not None:
            assert isinstance(v, ConversationBufferMemory)
            return v

        return ConversationBufferMemory(
            memory_key=values["memory_key"],
            return_messages=True,
            chat_memory=values["history"],
        )

    def converse(self, message: str) -> str:
        messages_template = [
            SystemMessagePromptTemplate.from_template(
                "You are Isabella, a chat agent at 'Magical World Hotel'."
                "Your job is to help customers book rooms and answer their questions."
                "You are very friendly, helpful, and respectful."
                "You are also very creative and clever."
                "You are a human."
            ),
            MessagesPlaceholder(variable_name=self.memory_key),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]

        formatted_conversation = self.format_conversation_with_incoming(message=message)

        predicted_intents = self.predict_intents(formatted_conversation)

        for i, intent in enumerate(predicted_intents, 1):
            data_collected = self.predict_gathered_data_for_itent(
                intent, formatted_conversation
            )

            try:
                parsed_info = self.parse_customer_information(
                    intent.action_parameters, data_collected
                )

                execute_intent = self.predic_check_for_intent_action(
                    intent=intent,
                    formatted_conversation=formatted_conversation,
                    customer_info=data_collected,
                )

                if execute_intent:
                    print(f"Executing intent: {intent.name}")
                    print(f"Calling action url: {intent.action_url}")
                    print(f"Action method: {intent.action_method}")
                    print(f"Action parameters: {intent.action_parameters}")
                    print(f"Parameters parsed: {parsed_info}")

                    # TODO: Send request to action url
                    # self.send_request_to_action_url(
                    #     intent.action_url, intent.action_method, parsed_info
                    # )
                    messages_template.insert(
                        i,
                        SystemMessagePromptTemplate.from_template(
                            f"We detected the intent: '''{intent.name}''' with description: '''{intent.description}'''."
                            "Don't ask the customer for more information."
                            "You already collected the information from the customer and executed the intent."
                            "Inform the customer that you executed the intent."
                            "You already collected the information from the customer and executed the intent."
                            "Inform the customer that you executed the intent."
                        ),
                    )
                else:
                    messages_template.insert(
                        i,
                        SystemMessagePromptTemplate.from_template(
                            f"We detected the intent: '''{intent.name}''' with description: '''{intent.description}'''."
                            "You already collected the information from the customer:\n"
                            f"'''\n{data_collected}\n'''\n"
                            "Ask the customer for confirmation with all the details you collected from them."
                            "Ask the customer to review the information and confirm if it's correct."
                        ),
                    )

            except ValueError as e:
                print(f"Error: {e}")

            messages_template.insert(
                i,
                SystemMessagePromptTemplate.from_template(
                    f"We detected the intent: '''{intent.name}''' with description: '''{intent.description}'''."
                    "Here's the information you should collect from the customer to fulfill the intent:\n"
                    f"'''\n{intent.action_parameters}\n'''\n"
                    "You can't execute the intent yet because you don't have all the information."
                    "You can't ask for more information because you don't know what information you need."
                    "Here's the information you already collected from the customer:\n"
                    f"'''\n{data_collected}\n'''\n"
                ),
            )

        chat_prompt_template = ChatPromptTemplate.from_messages(
            messages=messages_template
        )

        conversation_chain = ConversationChain(
            memory=self.memory,
            prompt=chat_prompt_template,
            llm=self.conversation_llm,
            verbose=self.verbose,
        )

        return conversation_chain.predict(input=message)

    def predict_intents(self, formatted_conversation: str) -> list[Intent]:
        intent_prompt_template = PromptTemplate.from_template(
            "Mark the intents that best matches the conversation with True or False.\n"
            "Output should follow the format: 'intent_name: True/False'\n"
            "All intents should be marked either True or False.\n"
            "Do not change the order or case of the intents.\n"
            "Do not add or remove any intents.\n"
            "Do not give any other information other than the intent name and True/False.\n"
            "Most recent messages should be considered first to determine the intent of the customer.\n"
            "Intents:\n{formatted_intents}\n\n"
            "Here is the conversation of the customer and the ai agent:\n"
            "'''{formatted_conversation}\n'''\n\n"
            "Intents:",
        )

        intent_chain = LLMChain(
            prompt=intent_prompt_template,
            llm=self.intent_llm,
            verbose=self.verbose,
        )

        intents_text = intent_chain.predict(
            formatted_intents=self.formatted_intents,
            formatted_conversation=formatted_conversation,
        )

        intents = self.parse_intents(intents_text=intents_text)
        return intents

    def predict_gathered_data_for_itent(
        self, itent: Intent, formatted_conversation: str
    ) -> str:
        gather_data_prompt = PromptTemplate.from_template(
            "Here is the information you should gather from the customer conversation to fulfill the intent '{intent_name}':\n"
            "'''\n{action_args}\n'''\n"
            "Here is the conversation of the customer and the ai agent:\n"
            "'''{formatted_conversation}\n'''\n\n"
            "Create a key-value pair for each argument in the format: 'key: value'\n"
            "All arguments should be provided.\n"
            "Do not change the order or case of the arguments.\n"
        )

        gather_data_chain = LLMChain(
            prompt=gather_data_prompt,
            llm=self.schema_llm,
            verbose=self.verbose,
        )

        gathered_data_text = gather_data_chain.predict(
            intent_name=itent.name,
            action_args=itent.action_parameters,
            formatted_conversation=formatted_conversation,
        )

        return gathered_data_text

    def predic_check_for_intent_action(
        self, intent: Intent, formatted_conversation: str, customer_info: str
    ) -> bool:
        action_confirmation_prompt = PromptTemplate.from_template(
            "Answer True if the ai agent already asked the customer to review the information provided and confirmed the action and the customer confirmed the action.\n"
            "Intent:\n'''\n{intent}\n'''\n"
            "Information collected:\n'''\n{customer_info}\n'''\n"
            "The lastest messagess are more relevant.\n"
            "Here is the conversation of the customer and the ai agent:\n"
            "'''{formatted_conversation}\n'''\n\n"
            "Output should follow the format: 'True/False'\n"
            "Confirmation should be True or False.\n"
        )

        action_confirmation_chain = LLMChain(
            prompt=action_confirmation_prompt,
            llm=self.confirmation_llm,
            verbose=self.verbose,
        )

        action_confirmation_text = action_confirmation_chain.predict(
            intent=intent.name,
            formatted_conversation=formatted_conversation,
            customer_info=customer_info,
        )

        action_confirmation = self.parse_action_confirmation(
            action_confirmation_text=action_confirmation_text
        )

        return action_confirmation

    def parse_intents(self, intents_text: str) -> list[Intent]:
        intents = [
            line.strip().split(": ") for line in intents_text.strip().split("\n")
        ]

        if not intents:
            return []

        active_intents = [name for name, active in intents if active == "True"]

        if not active_intents:
            return []

        return [intent for intent in self.intents if intent.name in active_intents]

    def parse_customer_information(
        self, action_params: list[ActionParameter], customer_info: str
    ) -> dict:
        parsed_info = {}

        for param in action_params:
            # Extract the field name and format from the ActionParameter object
            field_name = param.field
            field_format = param.format

            # Search for the corresponding field in the customer information
            search_string = f"{field_name}: "
            start_index = customer_info.find(search_string)
            if start_index == -1:
                # If the field is required but not found in the customer information, raise an exception
                if param.required:
                    raise ValueError(
                        f"Required field '{field_name}' not found in customer information"
                    )
                # If the field is not required and not found in the customer information, skip it
                else:
                    continue

            # Extract the value of the field from the customer information
            end_index = customer_info.find("\n", start_index)
            if end_index == -1:
                value = customer_info[start_index + len(search_string) :].strip()
            else:
                value = customer_info[
                    start_index + len(search_string) : end_index
                ].strip()

            # Remove any trailing non-numeric characters from the value if it's an integer
            if field_format == "integer":
                value = value.rstrip("'")
                value = int(value)
            # Validate the format of the value if specified in the ActionParameter object
            elif field_format == "float":
                value = value.rstrip("'")
                value = float(value)
            # Add the parsed value to the result dictionary
            parsed_info[field_name] = value

        return parsed_info

    def parse_action_confirmation(self, action_confirmation_text: str) -> bool:
        return action_confirmation_text.strip() == "True"

    @property
    def formatted_intents(self) -> str:
        return "\n".join(
            [f"- {intent.name} ({intent.description})" for intent in self.intents]
        )

    def format_conversation_with_incoming(self, message: str) -> str:
        messages = (
            f"{msg.get('type')}: {msg.get('data').get('content')}"
            for msg in messages_to_dict(self.history.messages)
        )

        conversation = "\n".join(messages)
        return f"{conversation}\nhuman: {message}"

    @classmethod
    def load_intents_from_json(self, path: str) -> list[Intent]:
        with open(path, "r") as f:
            data = json.load(f)

        intents = [
            Intent(
                name=intent.get("name"),
                description=intent.get("description"),
                action_url=intent.get("action_url"),
                action_method=intent.get("action_method"),
                action_parameters=intent.get("action_parameters"),
            )
            for intent in data.get("intents")
        ]

        return intents
