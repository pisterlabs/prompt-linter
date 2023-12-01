import logging
import json
import re
from dataclasses import asdict

# from py_backend.bots.AIBaseClass import AIFlatten
from py_backend.open_ai.openai_connection import (
    send_openai_functions_three,
    send_openai_functions_two,
)
from py_backend.storage.db_blueprint import load_blueprint_by_id
from py_backend.storage.db_objective import load_objective_by_id
from py_backend.bots.BlueprintClass import (
    BlueprintClass,
    ParameterSchema,
)
from py_backend.utils.kafka_conn import create_kafka_consumer, create_kafka_producer
from py_backend.utils.supabase import write_to_supabase

logging.getLogger("droid_assembly").setLevel(logging.WARNING)
log = logging.getLogger("droid_assembly")


def extract_json_objects(json_string):
    """Extract JSON objects from responses that get cut off"""
    pattern = r"\{[^{}]*\}"
    return [
        json.loads(match)
        for match in re.findall(pattern, json_string)
        if json.loads(match)
    ]


def is_valid_message_schema(payload):
    """Check if the message is valid"""
    if "source_id" not in payload:
        log.warning("Message does not have a source_id: %s", payload)
        return False

    if "payload" not in payload:
        log.warning("Payload is missing a payload: %s", payload)
        return False

    return True


def format_outbound_messages(source_id, source_type, data):
    """Prepare the outbound messages"""
    # TODO: Add a check to see if the data is None
    if data is None:
        log.warning(
            "%s: format_outbound_messages - data is None - %s",
            source_id,
            data,
        )
        return None

    try:
        res = [
            {
                "source_id": source_id,
                "source_type": source_type,
                "role": "assistant",
                "payload": data,
            }
        ]
        log.info(
            "%s: format_outbound_messages - %s",
            source_id,
            res,
        )
        return res
    except Exception as err:
        log.error(
            "%s: format_outbound_messages - error - %s",
            source_id,
            err,
        )


def flatten_messages(source_id, source_type, messages):
    res = []
    for msg in messages:
        formated_msg = format_outbound_messages(source_id, source_type, msg)
        if formated_msg:
            res.extend(formated_msg)
    return res


def update_roles(objects):
    """Replace the role of the AI bot with user"""
    return [
        {**obj, "role": "user"} if obj.get("role") == "ai_bot" else obj
        for obj in objects
    ]


class Droid(BlueprintClass):
    """Droid class for the basic droid assembly"""

    def __init__(
        self,
        blueprint_id,
        blueprint_name,
        droid_type,
        blueprint_description,
        initial_context,
        pub_topic_name=None,
        sub_topic_name=None,
        ignored_roles=None,
        message_counter=0,
        message_threshold=200,
    ):
        super().__init__(
            blueprint_id,
            blueprint_name,
            droid_type,
            blueprint_description,
            initial_context,
            ignored_roles,
            sub_topic_name,
            pub_topic_name,
        )
        if pub_topic_name is not None:
            self.pub_topic_name = pub_topic_name
        self.droid_type = droid_type
        self.source_id = self.blueprint_id
        self.message_counter = message_counter
        self.message_threshold = message_threshold
        self.consumer = create_kafka_consumer()
        self.producer = create_kafka_producer()

    def subscribe_to_topic(self):
        """Subscribe to the topic specified in the blueprint."""
        try:
            self.consumer.subscribe([self.sub_topic_name])
            print("Subscribed to topic: %s", self.sub_topic_name)
            log.info(
                "%s: subscribe_to_topic - success - %s",
                self.blueprint_name,
                self.sub_topic_name,
            )
        except Exception as err:
            log.error(
                "%s: subscribe_to_topic - error - %s - %s",
                self.blueprint_name,
                self.sub_topic_name,
                err,
            )

    def is_valid_inbound_source_id(self, payload):
        """Check if the inbound source_id is valid"""
        if "source_id" not in payload:
            log.warning("Message does not have a source_id: %s", payload)
            return False

        if payload["source_id"] == self.source_id:
            log.info("Message is from the same source_id, ignoring: %s", payload)
            return False

        return True

    def is_valid_inbound_role(self, payload):
        """Check if the inbound role is valid"""
        if "role" not in payload:
            log.warning("Payload is missing a role: %s", payload)
            return False

        if payload["role"] in self.ignored_roles:
            log.info(
                "Payload has an ignored role %s, ignoring: %s", payload["role"], payload
            )
            return False

        log.info("%s: is_valid_inbound_role - success - %s", self.source_id, payload)
        return True

    def is_valid_inbound_source_type(self, payload):
        """Check if the inbound source_type is valid"""

        if (
            "source_type" in payload
            and payload["source_type"] in self.ignored_source_types
        ):
            log.info(
                "Payload has an ignored source_type %s, ignoring: %s",
                payload["source_type"],
                payload,
            )
            return False

        log.info(
            "%s: is_valid_inbound_source_type - success - %s",
            self.source_id,
            payload,
        )
        return True

    def set_openai_query(self, inbound_message):
        """Get the OpenAI response"""
        _messages = []
        _messages.extend(self.initial_context)
        new_messages = [
            {
                "role": inbound_message["role"],
                "content": inbound_message["payload"],
            }
        ]
        _messages.extend(new_messages)
        return _messages

    def fetch_data(self, openai_query):
        """Get the OpenAI response"""
        try:
            # Sending the OpenAI query and getting the response
            response = send_openai_functions_three(messages=openai_query)

            # Writing the response to the database
            write_to_supabase(response, self.source_id)

            # Extracting the message content
            message_content = response["choices"][0]["message"]["content"]  # type: ignore

            # Logging the successful fetch
            log.info(
                "%s: fetch_data - return response from openai_query - %s",
                self.source_id,
                message_content,
            )
            return message_content
        except Exception as err:
            # Logging the error and returning None
            log.error(
                "%s: fetch_data - error - %s - %s",
                self.source_id,
                openai_query,
                err,
            )
            return None

    def send_messages_outbound(self, msgs):
        """Send messages to the Kafka topic"""
        log.info(
            "%s: send_messages_outbound - %s",
            self.source_id,
            self.pub_topic_name,
        )

        try:
            for msg in msgs:
                self.producer.send(self.pub_topic_name, value=msg)
                log.info(
                    "%s: send_messages_outbound - success - %s",
                    self.source_id,
                    msg,
                )

        except Exception as err:
            log.error(
                "%s: send_messages_outbound - error - %s - %s",
                self.source_id,
                self.pub_topic_name,
                err,
            )

    def run(self):
        """Run the AI"""

        self.subscribe_to_topic()

        for message in self.consumer:
            # handle_inbound_message
            if (is_valid_message_schema(message.value)) is False:
                continue
            if (self.is_valid_inbound_source_id(message.value)) is False:
                continue
            if (self.is_valid_inbound_role(message.value)) is False:
                continue
            if (self.is_valid_inbound_source_type(message.value)) is False:
                continue
            openai_query = self.set_openai_query(message.value)

            response = self.fetch_data(openai_query)

            if self.droid_type == "flatten":
                print(f"Flattening messages: {response} ")
                outbound_messages = flatten_messages(
                    source_id=self.source_id,
                    source_type=self.source_type,
                    messages=response,
                )

            else:
                outbound_messages = format_outbound_messages(
                    source_id=self.source_id,
                    source_type=self.source_type,
                    data=response,
                )
            print(f"outbound: {outbound_messages} ")

            self.send_messages_outbound(outbound_messages)

            # Self Counter
            self.message_counter += 1
            if self.message_counter > self.message_threshold:
                print("Unsubscribing from topic: %s", self.sub_topic_name)
                self.consumer.unsubscribe()
                break


class DroidFunction(Droid):
    """AI Parent Base Class"""

    def __init__(
        self,
        blueprint_id,
        blueprint_name,
        droid_type,
        blueprint_description,
        initial_context,
        functions,
        function_name,
        pub_topic_name=None,
        sub_topic_name=None,
        ignored_roles=None,
        message_counter=0,
        message_threshold=200,
    ):
        super().__init__(
            blueprint_id,
            blueprint_name,
            droid_type,
            blueprint_description,
            initial_context,
            ignored_roles,
            sub_topic_name,
            pub_topic_name,
        )
        if sub_topic_name is not None:
            self.sub_topic_name = sub_topic_name
        if pub_topic_name is not None:
            self.pub_topic_name = pub_topic_name
        self.droid_type = droid_type
        self.source_id = self.blueprint_id
        self.message_counter = message_counter
        self.message_threshold = message_threshold
        self.consumer = create_kafka_consumer()
        self.producer = create_kafka_producer()
        self.functions = functions
        self.function_name = function_name

    def fetch_data(self, openai_query):
        """Get the OpenAI response"""
        try:
            updated_openai_query = update_roles(openai_query)
            res = send_openai_functions_two(
                messages=updated_openai_query,
                functions=self.functions,
                function_name=self.function_name,
            )

            write_to_supabase(res, self.source_id)
            arguments_raw = res["choices"][0]["message"]["function_call"]["arguments"]  # type: ignore
            arguments = json.loads(arguments_raw)
            log.info(
                "%s: fetch_data - return response from openai_query - %s",
                self.source_id,
                arguments,
            )
            return arguments
        except Exception as error:
            log.error(
                "%s: fetch_data - error - %s - %s",
                self.source_id,
                openai_query,
                error,
            )

            return None


class DroidFlatten(Droid):
    """AI Parent Base Class

    * Recieve messages from a Kafka topic
    * Send messages to a OpenAI API
    * Recieve messages from a OpenAI API
    * Send messages to a Kafka topic

    """

    def __init__(
        self,
        blueprint_id,
        blueprint_name,
        droid_type,
        blueprint_description,
        initial_context,
        functions,
        function_name,
        pub_topic_name=None,
        sub_topic_name=None,
        ignored_roles=None,
        message_counter=0,
        message_threshold=200,
    ):
        super().__init__(
            blueprint_id,
            blueprint_name,
            droid_type,
            blueprint_description,
            initial_context,
            ignored_roles,
            sub_topic_name,
            pub_topic_name,
        )
        if sub_topic_name is not None:
            self.sub_topic_name = sub_topic_name
        if pub_topic_name is not None:
            self.pub_topic_name = pub_topic_name
        self.droid_type = droid_type
        self.source_id = self.blueprint_id
        self.message_counter = message_counter
        self.message_threshold = message_threshold
        self.consumer = create_kafka_consumer()
        self.producer = create_kafka_producer()
        self.functions = functions
        self.function_name = function_name

    def fetch_data(self, openai_query):
        """Get the OpenAI response"""
        try:
            updated_openai_query = update_roles(openai_query)
            res = send_openai_functions_two(
                messages=updated_openai_query,
                functions=self.functions,
                function_name=self.function_name,
            )

            write_to_supabase(res, self.source_id)
            function_call_raw = res["choices"][0].to_dict()["message"]["function_call"]  # type: ignore
            arguments = extract_json_objects(function_call_raw["arguments"])
            log.info(
                "%s: fetch_data - return response from openai_query - %s",
                self.source_id,
                arguments,
            )
            return arguments
        except Exception as error:
            log.error(
                "%s: fetch_data - error - %s - %s",
                self.source_id,
                openai_query,
                error,
            )

            return None


def load_objective(objective_id):
    """Load the blueprint and objective from the database by their IDs."""
    res_status_objective, objective = load_objective_by_id(objective_id)

    if not res_status_objective:
        error_msg = f"Failed to load objective with ID: {objective_id}"
        log.error(error_msg)
        raise ValueError(error_msg)

    return objective


def create_function_from_objective(objective):
    """Model and create the AI functions data scehma from the objective data"""
    # Check if required keys exist in the objective
    required_keys = ["parameters", "objective_name", "objective_description"]
    for key in required_keys:
        if key not in objective:
            logging.error(f"Key '{key}' missing in objective data.")
            return None

    try:
        par = asdict(ParameterSchema(objective["parameters"]))
        # print("par", par["properties"])
        print("asdasd", json.loads(objective["parameters"])["properties"])
        # fun = asdict(
        #     FunctionSchema(
        #         name=objective["objective_name"],
        #         description=objective["objective_description"],
        #         parameters=par,  # type: ignore
        #     )
        # )
        fun = {
            "name": objective["objective_name"],
            "description": objective["objective_description"],
            "parameters": {
                "type": "object",
                "properties": json.loads(objective["parameters"])["properties"],
            },
        }
        return fun

    except TypeError as err:
        logging.error(
            "Issue with the data format when creating function from objective: %s", err
        )
    except ValueError as err:
        logging.error("Invalid value provided in objective data: %s", err)
    except Exception as err:
        logging.error(
            "Unexpected error occurred while creating function from objective: %s", err
        )

    return None


def initialize_bot_basic(bp):
    """Initialize the bot based on the provided blueprint."""

    try:
        # Check if required attributes/keys are present in bp and fun
        # print(f"initialize_bot_basic: bp: {bp}")

        bot = Droid(
            blueprint_name=bp.blueprint_name,
            droid_type=bp.droid_type,
            blueprint_description=bp.blueprint_description,
            blueprint_id=bp.blueprint_id,
            initial_context=json.loads(bp.initial_context),
            pub_topic_name=bp.pub_topic_name,
        )

        print(f"initialize_bot_basic: bot: {bot}")

        return bot

    except (AttributeError, KeyError) as err:
        logging.error(err)
    except Exception as err:
        logging.error("Unexpected error occurred during AI bot initialization: %s", err)

    return None


def initialize_ai_bot_function(bp, fun):
    """Assign the droid to the right AIBaseClass method"""
    # print(f"initialize_ai_bot_flatten: bp: {bp.ignored_roles}")
    try:
        bot = DroidFunction(
            blueprint_name=bp.blueprint_name,
            droid_type=bp.droid_type,
            blueprint_description=bp.blueprint_description,
            blueprint_id=bp.blueprint_name,
            sub_topic_name=bp.sub_topic_name,
            pub_topic_name=bp.pub_topic_name,
            initial_context=json.loads(bp.initial_context),
            functions=[fun],
            function_name=fun["name"],
            # valid_schema=fun["parameters"],
            ignored_roles=["system"],
            # source_type="functional",
            # ignored_source_types=["functional"],
        )

        print(f"initialize_ai_bot_flatten: bot: {bot}")

        return bot

    except (AttributeError, KeyError) as err:
        logging.error(err)
    except Exception as err:
        logging.error("Unexpected error occurred during AI bot initialization: %s", err)

    return None


def initialize_ai_bot_flatten(bp, fun):
    """Assign the droid to the right AIBaseClass method"""
    # print(f"initialize_ai_bot_flatten: bp: {bp.ignored_roles}")
    try:
        bot = DroidFlatten(
            blueprint_name=bp.blueprint_name,
            droid_type=bp.droid_type,
            blueprint_description=bp.blueprint_description,
            blueprint_id=bp.blueprint_name,
            sub_topic_name=bp.sub_topic_name,
            pub_topic_name=bp.pub_topic_name,
            initial_context=json.loads(bp.initial_context),
            functions=[fun],
            function_name=fun["name"],
            # valid_schema=fun["parameters"],
            ignored_roles=["system"],
            # source_type="functional",
            # ignored_source_types=["functional"],
        )

        print(f"initialize_ai_bot_flatten: bot: {bot}")

        return bot

    except (AttributeError, KeyError) as err:
        logging.error(err)
    except Exception as err:
        logging.error("Unexpected error occurred during AI bot initialization: %s", err)

    return None


def run_droid_basic(blueprint_id, objective_id=None):
    """Run the droid based on the provided blueprint and objective IDs."""

    res_status_blueprint, blueprint = load_blueprint_by_id(blueprint_id)
    if not res_status_blueprint:
        error_msg = f"run_droid_basic: failed to load blueprint: {blueprint_id}"
        log.error(error_msg)
        raise ValueError(error_msg)

    if blueprint is None:
        logging.error("initialize_bot_basic: bp is None")
        return None
    if blueprint.droid_type is None:  # type: ignore
        logging.error("initialize_bot_basic: bp.droid_type is None")
        return None

    if blueprint.droid_type == "basic":  # type: ignore
        bot = initialize_bot_basic(blueprint)
        if bot:
            print(f"Starting AI: {bot.blueprint_name}")
            bot.run()
            print(f"Shutting down AI: {bot.blueprint_name}")

    if blueprint.droid_type == "function":  # type: ignore
        objective = load_objective(objective_id)
        function = create_function_from_objective(objective)
        bot = initialize_ai_bot_function(blueprint, function)
        if bot:
            print(f"Starting AI: {bot.source_id}")
            bot.run()
            print(f"Shutting down AI: {bot.source_id}")

    if blueprint.droid_type == "flatten":  # type: ignore
        objective = load_objective(objective_id)
        function = create_function_from_objective(objective)
        bot = initialize_ai_bot_flatten(blueprint, function)
        if bot:
            print(f"Starting AI: {bot.source_id}")
            bot.run()
            print(f"Shutting down AI: {bot.source_id}")
