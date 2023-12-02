# Tools that will be used by the chatbot

import json
import os
import traceback

from datetime import datetime, timedelta
from re import sub

from langchain import LLMChain
from langchain.agents import tool, create_sql_agent
from langchain.memory import ConversationBufferWindowMemory

from utils.constants import str_to_frequency, str_to_importance
from utils.create_goal_chain import get_create_goal_chain
from utils.goal_prompts import create_goal_chain_prompt
from utils.interactions import (
    create_goal,
    get_goal_info,
    retrieve_goal_pinecone,
    modify_goal,
    get_all_goals,
)
from utils.llm import BASE_CHATBOT_LLM
from utils.memory_utils import dict_to_memory
from utils.redis_user_data import (
    extend_user_msg_memory,
    get_user_hist,
    update_current_goal_creation_field_entries,
    reset_current_goal_creation_field_entries,
)
from utils.sms import send_sms

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
COHERE_EMBED_MODEL = "embed-english-light-v2.0"

##
# GOAL CREATION
# Conversational goal creation tool, enters a whole separate conversation
##

goal_create_memory = ConversationBufferWindowMemory(
    k=2, memory_key="history", input_key="input", ai_prefix="AI", human_prefix="User"
)

create_goal_chain = LLMChain(
    llm=BASE_CHATBOT_LLM,
    prompt=create_goal_chain_prompt,
    verbose=True,
    memory=goal_create_memory,
)


# Helper function to convert a string to camel case
def _camel_case(s):
    s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
    return "".join([s[0].lower(), s[1:]])


def parse_field_entries(field_entries: str):
    """
    Parses the raw output of the LLM field entries from a string to a dict

    The output keys for the dict will NOT be in camel case yet (use `format_text_fields` for this)
    """
    field_entries = field_entries.split("END FIELD ENTRIES")[0]

    return {
        field[0]: field[1]
        for field in [field.split(": ") for field in field_entries.split("\n")]
    }


def format_text_fields(fields: dict):
    """
    Formats the provided fields (from the parsed field entries) to prepare
    for the database

    Output goal_data:
      - name: string
      - description: string
      - estimatedImportance: 'HIGH' | 'MEDIUM' | 'LOW'
      - estimatedDurationHours: int
      - goalFrequency: 'DAILY' | 'WEEKLY' | None
      - reminderFrequency: 'HOURLY' | 'DAILY' | 'WEEKLY' | 'BIWEEKLY' | 'MONTHLY' | None
      - reminderTime: HH:MM
      - status: 'SUCCESS'
      - dueDate: datetime
      - isRecurring: 0 | 1
    """
    fields = fields.copy()
    if fields["Due Date Year"] == "N/A":
        fields["Due Date"] = None
    else:
        fields["Due Date"] = datetime(
            int(fields["Due Date Year"]),
            int(fields["Due Date Month"]),
            int(fields["Due Date Day"]),
            int(0 if fields["Due Date Hour"] == "N/A" else fields["Due Date Hour"]),
            int(0 if fields["Due Date Minute"] == "N/A" else fields["Due Date Minute"]),
        )
    fields["Is Recurring"] = (int)(fields["Goal Type"] == "RECURRING")

    for key in [
        "Due Date Year",
        "Due Date Month",
        "Due Date Day",
        "Due Date Hour",
        "Due Date Minute",
        "Goal Type",
    ]:
        del fields[key]

    return {
        _camel_case(key): value if value != "N/A" else None
        for key, value in fields.items()
    }


def prettify_field_entries(fields: dict):
    """
    Takes the parsed field entries (in dict form, just not formatted with camelCase)
    and prettifies them to display to the user
    """

    data = format_text_fields(fields)
    pretty_output = f'🎯 {data["name"]}' + "\n\n"
    pretty_output += data["description"] + "\n\n"
    if data["dueDate"]:
        pretty_output += (
            f'📆 Due date: {data["dueDate"].strftime("%m/%d/%Y, %H:%M:%S")}' + "\n"
        )
    if data["estimatedImportance"]:
        pretty_output += (
            f'⭐️ Priority level: {data["estimatedImportance"].lower()}' + "\n"
        )
    if data["estimatedDurationHours"]:
        pretty_output += (
            f'⏳ Estimated duration: {data["estimatedDurationHours"]} hours' + "\n"
        )
    if data["goalFrequency"]:
        pretty_output += f'💪 Goal frequency: {data["goalFrequency"].lower()}' + "\n"
    pretty_output += (
        f'{"🔁" if data["isRecurring"] else "⤴️"} Goal type: {"recurring" if data["isRecurring"] else "one-time"}'
        + "\n"
    )
    pretty_output += (
        f'⏲️ Reminder frequency: {data["reminderFrequency"].lower() if data["reminderFrequency"] else "N/A"}'
        + "\n"
    )
    pretty_output += f'⏰ Reminder time: {data["reminderTime"]}'

    return pretty_output


# Function that RETURNS a user-specific tool for creating a goal
# def init_conversational_create_goal_tool(user: str) -> callable:
#     def conversational_create_goal_tool(query: str) -> str:
#         """
#         A tool which may prompt for additional user input to aid for the creation of a user goal.
#         Upon running this tool, a new conversation will be started with a separate model, and info
#         will be updated accordingly. The output of this model is simply the first response from the
#         chain. Memory will be saved, and the conversation type will be updated.
#         """

#         user_input = query
#         current_field_entries = None

#         # If this tool is being run, we can optionally alert the user that we're working
#         # on adding a goal for them (so they know that the model is "thinking")
#         send_sms(user, "Okay, I'm working on designing a goal for you!")

#         # Set current convo type to create goal
#         update_user_convo_type(user, "create_goal")

#         # Create chain
#         chain, memory = init_create_goal_chain(DEBUG=True)

#         # Make prediction
#         current_full_output = chain.predict(input=user_input, today=datetime.now())

#         # Extract field entries and output
#         print(current_full_output)
#         current_field_entries = parse_field_entries(
#             current_full_output.split("END FIELD ENTRIES")[0].strip()
#         )
#         current_conversational_output = current_full_output.split("GoalDesigner: ")[
#             1
#         ].strip()

#         # Save memory of this conversation
#         extend_user_msg_memory(user, "create_goal", memory_to_dict(memory))

#         # This shouldn't happen on the first round (because the model was "told not to")
#         # but just in case
#         if current_field_entries["STATUS"] == "SUCCESS":
#             assert False
#             update_user_convo_type(user, "main")

#             # Parse current field entries here
#             # and add them to the database
#             fields_json = text_fields_to_json(current_field_entries)

#             # TODO: add embedding using pgvector
#             goal_name_embedding = create_embedding(current_field_entries["name"])
#             create_goal(fields_json)

#             return f"The goal data being added is as follows:\n{current_field_entries}\nGoal added successfully!"

#         # This output will be used directly
#         pretty_field_entries = prettify_field_entries(current_field_entries)
#         return f"{pretty_field_entries}\n\n{current_conversational_output}"

#     return conversational_create_goal_tool


def init_create_goal_tool_ALT(user: str) -> callable:
    def create_goal_tool(query: str) -> str:
        print(">>> CALLED create_goal_tool")
        """
        A tool that should be called whenever the bot needs to respond in a fashion related
        to creating a goal.
        """

        user_data = get_user_hist(user)
        user_input = query
        current_field_entries = None

        if user_data["current_field_entries"] != {}:
            return "You are already in the process of creating a goal!"

        # If this tool is being run, we can optionally alert the user that we're working
        # on adding a goal for them (so they know that the model is "thinking")
        send_sms(user, "Okay, I'm working on designing a goal for you!")
        extend_user_msg_memory(
            user,
            "main",
            [
                {
                    "type": "ai",
                    "data": {
                        "content": "Okay, I'm working on designing a goal for you!",
                        "additional_kwargs": {},
                        "example": False,
                    },
                }
            ],
        )

        # Load memory
        create_memory = dict_to_memory(user_data["main_memory"])

        # Load chain for goal creation conversation
        chain = get_create_goal_chain(create_memory, DEBUG=True)

        # Make prediction
        current_full_output = chain.predict(input=user_input, today=datetime.now())

        # Extract field entries and output
        print(current_full_output)
        current_field_entries = parse_field_entries(
            current_full_output.split("END FIELD ENTRIES")[0].strip()
        )
        current_conversational_output = current_full_output.split("GoalDesigner: ")[
            1
        ].strip()

        # Store field entries info locally
        update_current_goal_creation_field_entries(
            user, current_field_entries, datetime.now()
        )

        # If the status is marked as success, then we shouldn't have to call
        # the tool again until the next time the user wants to create a goal
        if current_field_entries["STATUS"] == "SUCCESS":
            # Parse current field entries here
            # and add them to the database
            formatted_text_fields = format_text_fields(current_field_entries)
            create_goal(formatted_text_fields, user)
            reset_current_goal_creation_field_entries(user)
            return "Goal successfully created!"

        # This output will be used directly
        pretty_field_entries = prettify_field_entries(current_field_entries)
        return f"{pretty_field_entries}\n\n{current_conversational_output}"

    return create_goal_tool


def init_create_goal_modify_tool_ALT(user: str) -> callable:
    print(">>> CALLED init_create_goal_modify_tool_ALT")

    def create_goal_modify_tool(query: str) -> str:
        """
        A tool that should be called whenever the bot needs to respond in a fashion related
        to creating a goal.
        """
        user_data = get_user_hist(user)
        user_input = query
        current_field_entries = None

        # check if timestamp on message is valid
        if datetime.strptime(
            user_data["last_user_message_time"], "%m/%d/%Y %H:%M:%S"
        ) < datetime.strptime(
            user_data["current_field_entries_last_modified"], "%m/%d/%Y %H:%M:%S"
        ):
            return "You are already in the process of creating a goal!"

        # If this tool is being run, we can optionally alert the user that we're working
        # on adding a goal for them (so they know that the model is "thinking")
        send_sms(
            user, "Just a moment, I'm working on changing this new goal appropriately."
        )
        extend_user_msg_memory(
            user,
            "main",
            [
                {
                    "type": "ai",
                    "data": {
                        "content": "Just a moment, I'm working on changing this new goal appropriately.",
                        "additional_kwargs": {},
                        "example": False,
                    },
                }
            ],
        )

        # Load memory
        create_memory = dict_to_memory(user_data["main_memory"])

        # Load chain for goal creation conversation
        chain = get_create_goal_chain(create_memory, DEBUG=True)

        # Make prediction
        current_full_output = chain.predict(input=user_input, today=datetime.now())

        # Extract field entries and output
        current_field_entries = parse_field_entries(
            current_full_output.split("END FIELD ENTRIES")[0].strip()
        )
        current_conversational_output = current_full_output.split("GoalDesigner: ")[
            1
        ].strip()

        # Store field entries info locally
        update_current_goal_creation_field_entries(
            user, current_field_entries, datetime.now()
        )

        # If the status is marked as success, then we shouldn't have to call
        # the tool again until the next time the user wants to create a goal
        if current_field_entries["STATUS"] == "SUCCESS":
            # Parse current field entries here
            # and add them to the database
            formatted_text_fields = format_text_fields(current_field_entries)
            create_goal(formatted_text_fields, user)
            reset_current_goal_creation_field_entries(user)
            return "Goal successfully created!"

        # This output will be used directly
        pretty_field_entries = prettify_field_entries(current_field_entries)
        return f"{pretty_field_entries}\n\n{current_conversational_output}"

    return create_goal_modify_tool


def init_create_goal_finish_tool_ALT(user: str) -> callable:
    print(">>> CALLED init_create_goal_finish_tool_ALT")

    def create_goal_finish_tool(query: str) -> str:
        # Need to get the most updated version of user data
        user_data = get_user_hist(user)
        field_entries = user_data["current_field_entries"]

        # check if timestamp on message is valid
        if datetime.strptime(
            user_data["last_user_message_time"], "%m/%d/%Y %H:%M:%S"
        ) < datetime.strptime(
            user_data["current_field_entries_last_modified"], "%m/%d/%Y %H:%M:%S"
        ):
            return "You are already in the process of creating a goal!"

        send_sms(user, "Ok, I'm saving your goal!")
        extend_user_msg_memory(
            user,
            "main",
            [
                {
                    "type": "ai",
                    "data": {
                        "content": "Ok, I'm saving your goal!",
                        "additional_kwargs": {},
                        "example": False,
                    },
                }
            ],
        )

        if field_entries:
            formatted_text_fields = format_text_fields(field_entries)
            create_goal(formatted_text_fields, user)

            # Reset
            reset_current_goal_creation_field_entries(user)
            return "Goal successfully created!"
        else:
            return "Failed to upload goal, as the user was not currently in the process of creating a goal."

    return create_goal_finish_tool


##
# GOAL READING/EDITING
# Available tools:
# - Summarize goal recent activity
# - Ask question about goal
# - Get goal specific info (database)
##


def init_get_specific_goal_tool(user: str) -> callable:
    def get_specific_goal_tool(goal_name_query: str) -> str:
        """
        A tool used to retrieve info about a specific goal.
        """
        goal_id = retrieve_goal_pinecone(goal_name_query, user[1:])

        if not goal_id:
            return "The user does not appear to have any goals related to this query."

        return get_goal_info(goal_id)

    return get_specific_goal_tool


def init_get_all_goals_tool(user: str) -> callable:
    # LangChain requires an input here even if we aren't using it
    def get_all_goals_tool(query: str) -> str:
        return get_all_goals()

    return get_all_goals_tool


def init_modify_specific_goal_tool(user: str) -> callable:
    def modify_specific_goal_tool(query: str) -> str:
        """
        A tool used to edit information about a specific goal.
        """

        print(query)

        goal_name_query, modifications = query.split(": ", 1)
        modifications = json.loads(modifications)
        if "reminder_start_time" in modifications.keys():
            new_time = datetime.strptime(
                modifications["reminder_start_time"], "%H:%M:%S"
            ).time()
            print(datetime, timedelta)
            tomorrow = datetime.now() + timedelta(days=1)
            modifications["reminder_start_time"] = datetime.combine(tomorrow, new_time)
        if "frequency" in modifications.keys():
            modifications["frequency"] = str_to_frequency.get(
                modifications["frequency"], 0
            )
        if "importance" in modifications.keys():
            modifications["importance"] = str_to_importance.get(
                modifications["importance"], 0
            )

        print(modifications)

        # Retrieve goal by query
        goal_id = retrieve_goal_pinecone(goal_name_query, user[1:])

        # Update goal fields
        try:
            modify_goal(goal_id, modifications)
            return "Goal modified successfully!"
        except Exception:
            print(traceback.format_exc())
            # print(e)
            return "Error occurred when modifying goal."

    return modify_specific_goal_tool


# goal_db_toolkit = SQLDatabaseToolkit(db=goal_database)
# goal_db_agent_executor = create_sql_agent(
#     llm=OpenAI(temperature=0),
#     toolkit=goal_db_toolkit,
#     prefix=GOAL_DB_PREFIX,
#     verbose=True,
# )
# goal_db_reader_tool = goal_db_agent_executor.run

# @tool
# def goal_db_reader_
