import logging
import os
import random
from typing import Dict, Text, Any, List, Optional

from rasa_sdk import Action
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher

from actions.api_llm.llms import OpenAI
from actions.api_llm.prompts import OpenAIPrompts

DEFAULT_REPHRASE_RESPONSE = os.environ.get(
    "DEFAULT_REPHRASE_RESPONSE", "utter_ask_rephrase"
)

logger = logging.getLogger(__name__)


def select_a_responses(
    domain: Dict[Text, Any], name: Text, channel: Text
) -> Optional[Text]:
    """
    Retrieves responses, filters by channel, and
    selects a random response among candidate responses
    """
    # retrieve responses
    responses = domain.get("responses", {}).get(name, [])

    # selecting the response at random
    # and based on the input channel
    responses = [
        response for response in responses if response.get("channel") in [None, channel]
    ]
    if not responses:
        return None

    response = random.choice(responses) if len(responses) > 1 else responses[0]
    return response.get("text", None)


# First two custom actions are generic and
# can be used with intents where there is a
# matching response name. For example, to use
# these actions with greet, there must be an
# utter_greet response.
class ActionLLMGenerateResponseCommon(Action):
    """
    Trigger LLM API call to generate a specified response.
    To use this action, developers must ensure that there
    is a response with the utter_ prefix along with the name
    of the previous intent. This action then retrieves the
    response and generates a new response using a specified LLM.

    In addition, this action is capable of utilizing custom
    prompts based on the name of the intent, and fallback to
    the default prompt if a prompt is not specified in the
    llm_prompts.yml file.
    """

    def name(self) -> Text:
        return "action_llm_generate_response_common"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        latest_intent = tracker.get_intent_of_latest_message()
        latest_query = tracker.latest_message.get("text", None)
        input_channel = tracker.get_latest_input_channel()

        if not latest_query or not latest_intent:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.info(f"Could not retrieve the latest intent or user query")
            return []

        # retrieve responses for the input channel
        response = select_a_responses(
            domain=domain, name=f"utter_{latest_intent}", channel=input_channel
        )
        if not response:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.info(
                f"Could not retrieve the response for intent: utter_{latest_intent}"
            )
            return []

        llm = OpenAI()
        prompt = OpenAIPrompts.get_generate_prompt(
            template=f"utter_{latest_intent}",
            query=latest_query,
            context=response,
            configure_personality=True,
        )

        try:
            llm_response = llm.get_text_completion(prompt=prompt)

            if llm_response:
                dispatcher.utter_message(text=llm_response)
                logger.info(f"LLM response generated")
            else:
                dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
                logger.info(
                    f"LLM returned an empty response. Falling back to the rephrase utterance"
                )
        except Exception as e:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.exception(
                f"An exception occurred while generating the LLM response. "
                f"Falling back to the rephrase utterance. {e}"
            )
        return []


class ActionLLMRephraseResponseCommon(Action):
    """
    Trigger LLM API call to rephrase a specified response.
    To use this action, developers must ensure that there
    is a response with the utter_ prefix along with the name
    of the previous intent. This action then retrieves the
    response and rephrases it using a specified LLM.

    In addition, this action is capable of utilizing custom
    prompts based on the name of the intent, and fallback to
    the default prompt if a prompt is not specified in the
    llm_prompts.yml file.
    """

    def name(self) -> Text:
        return "action_llm_rephrase_response_common"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        latest_intent = tracker.get_intent_of_latest_message()
        latest_query = tracker.latest_message.get("text", None)
        input_channel = tracker.get_latest_input_channel()

        if not latest_query or not latest_intent:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.info(f"Could not retrieve the latest intent or user query")
            return []

        # retrieve responses for the input channel
        response = select_a_responses(
            domain=domain, name=f"utter_{latest_intent}", channel=input_channel
        )
        if not response:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.info(
                f"Could not retrieve the response for intent: `utter_{latest_intent}`"
            )
            return []

        llm = OpenAI()
        prompt = OpenAIPrompts.get_rephrase_prompt(
            template=f"utter_{latest_intent}",
            query=latest_query,
            response=response,
            configure_personality=True,
        )

        try:
            llm_response = llm.get_text_completion(prompt=prompt)

            if llm_response:
                dispatcher.utter_message(text=llm_response)
                logger.info(f"LLM response generated")
            else:
                dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
                logger.info(
                    f"LLM returned an empty response. Falling back to the rephrase utterance"
                )
        except Exception as e:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.exception(
                f"An exception occurred while generating the LLM response. "
                f"Falling back to the rephrase utterance. {e}"
            )
        return []


# The following custom actions are specific,
# and they can be used when a custom prompt
# must be triggerred. Also, there are some use
# cases where above generic LLM actions cannot
# be used, such as within stories, where a
# dedicated action must be written.
class ActionLLMRephraseResponseWelcome(Action):
    """
    A specific LLM response generation action
    for welcoming the users for the first time.
    """

    def name(self) -> Text:
        return "action_llm_rephrase_response_welcome"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # define a custom prompt template
        # and placeholders to replace
        template_name = "welcome"

        llm = OpenAI()
        prompt = OpenAIPrompts.get_custom_prompt(
            template_name=template_name,
            configure_personality=True,
        )

        try:
            llm_response = llm.get_text_completion(prompt=prompt)

            if llm_response:
                dispatcher.utter_message(text=llm_response)
                logger.info(f"LLM response generated")
            else:
                dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
                logger.info(
                    f"LLM returned an empty response. Falling back to the rephrase utterance"
                )
        except Exception as e:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.exception(
                f"An exception occurred while generating the LLM response. "
                f"Falling back to the rephrase utterance. {e}"
            )
        return []


class ActionLLMGenerateResponseMenu(Action):
    """
    A specific LLM response generation action
    for menu user intent (for stories).
    """

    def name(self) -> Text:
        return "action_llm_generate_response_menu"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        latest_query = tracker.latest_message.get("text", None)
        input_channel = tracker.get_latest_input_channel()

        if not latest_query:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.info(f"Could not retrieve the latest  user query")
            return []

        # get utter_menu response
        response = select_a_responses(
            domain=domain, name="utter_menu", channel=input_channel
        )
        if not response:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.info(f"Could not retrieve `utter_menu`")
            return []

        llm = OpenAI()
        prompt = OpenAIPrompts.get_generate_prompt(
            template="utter_menu",
            query=latest_query,
            context=response,
            configure_personality=True,
        )

        try:
            llm_response = llm.get_text_completion(prompt=prompt)

            if llm_response:
                dispatcher.utter_message(text=llm_response)
                logger.info(f"LLM response generated")
            else:
                dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
                logger.info(
                    f"LLM returned an empty response. Falling back to the rephrase utterance"
                )
        except Exception as e:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.exception(
                f"An exception occurred while generating the LLM response. "
                f"Falling back to the rephrase utterance. {e}"
            )
        return []


class ActionLLMRephraseResponseGoodbye(Action):
    """
    A specific LLM response generation action
    for rephrasing goodbye utterance (for stories).
    """

    def name(self) -> Text:
        return "action_llm_rephrase_response_goodbye"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        latest_query = tracker.latest_message.get("text", None)
        input_channel = tracker.get_latest_input_channel()

        if not latest_query:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.info(f"Could not retrieve the latest  user query")
            return []

        # get utter_menu response
        response = select_a_responses(
            domain=domain, name="utter_goodbye", channel=input_channel
        )
        if not response:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.info(f"Could not retrieve `utter_goodbye`")
            return []

        llm = OpenAI()
        prompt = OpenAIPrompts.get_rephrase_prompt(
            template="utter_goodbye",
            query=latest_query,
            response=response,
            configure_personality=True,
        )

        try:
            llm_response = llm.get_text_completion(prompt=prompt)

            if llm_response:
                dispatcher.utter_message(text=llm_response)
                logger.info(f"LLM response generated")
            else:
                dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
                logger.info(
                    f"LLM returned an empty response. Falling back to the rephrase utterance"
                )
        except Exception as e:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.exception(
                f"An exception occurred while generating the LLM response. "
                f"Falling back to the rephrase utterance. {e}"
            )
        return []


class ActionLLMGenerateResponsePrices(Action):
    """
    A specific LLM response generation action
    for prices related questions since the domain
    does not have an utter_prices response which
    prevents us from using the common LLM response
    generation or rephrase actions.
    """

    def name(self) -> Text:
        return "action_llm_generate_response_prices"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        latest_query = tracker.latest_message.get("text", None)
        input_channel = tracker.get_latest_input_channel()

        if not latest_query:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.info(f"Could not retrieve the latest  user query")
            return []

        # pricing data is already available in
        # the utter_menu response
        pricing = select_a_responses(
            domain=domain, name="utter_menu", channel=input_channel
        )
        if not pricing:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.info(f"Could not retrieve the pricing list from `utter_menu`")
            return []

        # define a custom prompt template
        # and placeholders to replace
        template_name = "prices"
        placeholders = {"pricing": pricing, "query": latest_query}

        llm = OpenAI()
        prompt = OpenAIPrompts.get_custom_prompt(
            template_name=template_name,
            placeholders=placeholders,
            configure_personality=True,
        )

        try:
            llm_response = llm.get_text_completion(prompt=prompt)

            if llm_response:
                dispatcher.utter_message(text=llm_response)
                logger.info(f"LLM response generated")
            else:
                dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
                logger.info(
                    f"LLM returned an empty response. Falling back to the rephrase utterance"
                )
        except Exception as e:
            dispatcher.utter_message(template=DEFAULT_REPHRASE_RESPONSE)
            logger.exception(
                f"An exception occurred while generating the LLM response. "
                f"Falling back to the rephrase utterance. {e}"
            )
        return []
