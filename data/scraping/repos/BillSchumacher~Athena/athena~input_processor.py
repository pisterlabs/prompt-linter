from loguru import logger

import plugins
from athena.llm.openai.completion import openai_completion
from athena.plugins.authentication_plugin import AuthenticationPlugin
from athena.plugins.plugin_base import PluginBase
from athena.plugins.plugin_manager import PluginManager
from athena.prompt import SYSTEM_PROMPT
from athena.user_manager import UserManager

user_manager = UserManager()
auth_plugin = AuthenticationPlugin(user_manager)


def process_input(
    intent_pipelein,
    entity_pipeline,
    user_input,
    username=None,
    completion_callback=None,
):
    if not completion_callback:
        completion_callback = openai_completion
    logger.info(f"User input received: {user_input}")
    if user_input == "":
        return "I'm sorry, I didn't receive any input. Can you please try again?"
    # Use the Rasa NLU interpreter to parse the user input
    parsed_input = intent_pipelein(user_input)
    logger.debug(f"Parsed input: {parsed_input}")

    # Extract intent and entities from the parsed input
    intent, intent_confidence = parsed_input
    entities = entity_pipeline(user_input)
    logger.debug(
        f"Parsed intent: {intent} confidence: {intent_confidence} entities: {entities}"
    )

    # Create a PluginManager instance, discover the plugins, and process the input using the plugins
    plugin_manager = PluginManager(PluginBase, plugins, extra_plugins=[auth_plugin])
    plugin_manager.discover_plugins()

    if username is not None:
        logger.debug(f"User {username} is logged in. Applying personalization...")
        personalization_data = user_manager.get_user_preference(
            username, "personalization_data"
        )
        if personalization_data is not None:
            logger.debug(f"Personalization data found: {personalization_data}")
            # Apply personalization logic using personalization_data
        else:
            logger.debug("No personalization data found.")

    response = plugin_manager.process_input(user_input)
    prompt = f"{SYSTEM_PROMPT}{user_input}"
    if response is None and intent_confidence < 0.65:
        logger.debug(
            "No plugin was able to process the input and the intent confidence is low. Using GPT-3 to generate a response."
        )
        response = completion_callback(prompt)

    if response is None:
        logger.debug("No plugin was able to process the input. Using default logic.")
        if intent == "greeting":
            response = "Hello! How can I help you today?"
        elif intent == "goodbye":
            response = "Goodbye! Have a great day!"
        elif intent == "current_state":
            response = "I've been busy!"
        elif intent == "name":
            response = "My name is Athena!"
        elif intent == "weather":
            location = "unknown"
            if entities:
                for entity in entities:
                    if entity[0] == "GPE":
                        location = entities[0][-1]
            response = f"I'm not currently able to check the weather, but you asked about {location}."
        else:
            logger.debug("No intent was detected. Using GPT-3 to generate a response.")

            response = completion_callback(prompt)
    logger.info(f"Response generated: {response}")
    return response
