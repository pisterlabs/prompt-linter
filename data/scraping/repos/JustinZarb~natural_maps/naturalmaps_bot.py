import openai
import os
import json
import pandas as pd
import requests
from time import localtime, strftime
import osmnx as ox


class ChatBot:
    def __init__(self, log_path: str = None):
        # Get OpenAI Key
        self.get_openai_key_from_env()
        assert openai.api_key, "Failed to find API keys"

        # Initialize Messages
        self.messages = []

        # Invalid messages cannot be added to the chat but should be saved For logging
        self.invalid_messages = []

        # Initialize Functions
        self.functions = {
            "" "overpass_query": self.overpass_query,
        }
        self.function_status_pass = False  # Used to indicate function success
        self.function_metadata = [
            {
                "name": "overpass_query",
                "description": """Run an overpass QL query.
                    Instructions:
                    - Keep the queries simple and specific.
                    - Always use Overpass built-in geocodeArea for locations like this /{/{geocodeArea:charlottenburg}}->.searchArea; 
                    - Do not exceed size to 100 unless a previous attempt was unsuccessful.
                    - If running broad searches such as [node[~'^(amenity|leisure)$'~'.'](\{\{bbox}});], stick to only nodes. 
                    eg. prompt: "Find toilets in Charlottenburg"
                    
                    [out:json][timeout:25];
                    /{/{geocodeArea:charlottenburg}}->.searchArea;
                    (
                    node["amenity"="toilets"](area.searchArea);
                    way["amenity"="toilets"](area.searchArea);
                    relation["amenity"="toilets"](area.searchArea);
                    );
                    out body;
                    >;
                    out skel qt;
                    """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "human_prompt": {
                            "type": "string",
                            "description": "The user message. Used for logging. Do not paraphrase.",
                        },
                        "generated_query": {
                            "type": "string",
                            "description": "The overpass QL query to execute. Important: Ensure that this is a properly formatted .json string.",
                        },
                    },
                    "required": ["prompt", "query"],
                },
            },
        ]

        # Logging parameters
        self.overpass_queries = {}
        self.id = self.get_timestamp()
        if log_path is None:
            log_path = "~/naturalmaps_logs"
            self.log_path = os.path.expanduser(log_path)

    def name_to_gdf(self, place_name):
        # Use OSMnx to geocode the location
        return ox.geocode_to_gdf(place_name)

    def save_to_json(self, file_path: str, this_run_name: str, log: dict):
        json_file_path = file_path

        # Check if the folder exists and if not, create it.
        folder_path = os.path.dirname(json_file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Check if the file exists
        if os.path.isfile(json_file_path):
            # If it exists, open it and load the JSON data
            with open(json_file_path, "r") as f:
                data = json.load(f)
        else:
            # If it doesn't exist, create an empty dictionary
            data = {}

        # Add data for this run
        data[this_run_name] = {
            "log": log,
        }

        with open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)

    def get_timestamp(self):
        return strftime("%Y-%m-%d %H:%M:%S", localtime())

    def get_openai_key_from_env(self):
        # Get api_key (saved locally)
        api_key = os.getenv("OPENAI_KEY")
        openai.api_key = api_key

    def log_overpass_query(self, human_prompt, generated_query, data_str):
        # Write Overpass API Call to JSON
        timestamp = self.get_timestamp()
        this_run_name = f"{timestamp} | {human_prompt}"
        filepath = os.path.join(self.log_path, "overpass_query_log.json")
        success = True if "error" not in data_str else False
        data_dict = json.loads(data_str)
        returned_something = (
            True
            if ("elements" in data_dict and len(data_dict["elements"])) > 0
            else False
        )

        # This gets saved in the chat log
        self.overpass_queries[generated_query] = {
            "valid_query": success,
            "returned_something": returned_something,
            "data": data_str,
        }

        # This gets saved in a separate log for overpass ueries
        self.save_to_json(
            file_path=filepath,
            this_run_name=this_run_name,
            log={
                "overpassql_query": generated_query,
                "overpass_response": data_str,
                "valid_query": success,
                "returned_something": returned_something,
            },
        )

    def overpass_query(self, human_prompt, generated_query):
        """Run an overpass query
        To improve chances of success, run this multiple times for simpler queries.
        eg. prompt: "Find bike parking near tech parks in Kreuzberg, Berlin"
        in this example, a complex query is likely to fail, so it is better to run
        a first query for bike parking in Kreuzberk and a second one for tech parks in Kreuzberg
        """
        overpass_url = "http://overpass-api.de/api/interpreter"
        response = requests.get(overpass_url, params={"data": generated_query})
        if response.content:
            try:
                data = response.json()
            except:
                data = {"error": str(response)}
        else:  # ToDo: check this.. it might not make sense
            data = {
                "warning": "received an empty response from Overpass API. Tell the user."
            }
        data_str = json.dumps(data)
        self.log_overpass_query(human_prompt, generated_query, data_str)
        return data_str

    def add_system_message(self, content):
        self.messages.append({"role": "system", "content": content})

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})

        self.add_system_message(
            content="""Let's first understand the problem and devise a plan to solve the problem."
            " Please output the plan starting with the header 'Plan:' "
            "and then followed by a numbered list of steps. "
            "Please make the plan the minimum number of steps required "
            "to accurately complete the task. If the task is a question, "
            "the final step should almost always be 'Given the above steps taken, "
            "please respond to the users original question'. "
            "At the end of your plan, say '<END_OF_PLAN>'"""
        )

    def add_function_message(self, function_name, content):
        self.messages.append(
            {"role": "function", "name": function_name, "content": content}
        )

    def execute_function(self, response_message):
        """Execute a function from self.functions

        Args:
            response_message (_type_): The message from the language model with the required inputs
            to run the function
        """
        # Return false if we decide that the function failed
        self.function_status_pass = False

        function_name = response_message["function_call"]["name"]
        function_args = response_message["function_call"]["arguments"]

        if function_name in self.functions:
            function_to_call = self.functions[function_name]
            try:
                function_args_dict = json.loads(function_args)
                json_failed = False
            except json.JSONDecodeError as e:
                json_failed = True
                function_response = {
                    "invalid args": str(e),
                    "input": function_args,
                }

            if not json_failed:
                try:
                    function_response = function_to_call(**function_args_dict)
                except TypeError as e:
                    function_response = {
                        "invalid args": str(e),
                        "input": function_args,
                    }
                # Specific checks for self.overpass_query()
                if function_name == "overpass_query":
                    data = json.loads(function_response)
                    if len(function_response) > 4096:
                        function_response = "Overpass query returned too many results."
                    if "elements" in data:
                        elements = data["elements"]
                        if elements == []:
                            function_response += (
                                "-> Overpass query returned no results."
                            )
                        else:
                            # Overpass query worked! Passed!
                            self.function_status_pass = True

        else:
            function_response = f"{function_name} not found"

        self.add_function_message(function_name, function_response)
        self.add_system_message(
            content=f"Treat each of the steps in the plan as discreet steps. Does the function response contain enough information"
            f"to answer step {self.current_step}? If not, state what you would do next (eg. 'trying again with a larger search area')."
            f"If it did, respond with 'Step {self.current_step+1}' and solve this step."
            "If you do not have an adequate function to run the next step, give an appropriate message and skip to the final step."
            "If some steps failed, provide a response explaining what worked and what didn't. Provide any useful information from partial results."
        )

    def is_valid_message(self, message):
        """Check if the message content is a valid JSON string"""
        if message.get("function_call"):
            if message["function_call"].get("arguments"):
                function_args = message["function_call"]["arguments"]
                if isinstance(function_args, str):
                    try:
                        json.loads(function_args)  # Attempt to parse the JSON string
                        return True  # Return True if it's a valid JSON string
                    except json.JSONDecodeError:
                        return False  # Return False if it's not a valid JSON string
            return False
        else:
            return True

    def process_messages(self, n=1):
        """A general purpose function to prepare an answer based on all the previous messages

        Issues: currently modifying the original prompt

        Args:
            n (int, optional): Changes the number of responses from GPT.
            Increasing this raises the chances tha one answer will generate
            a valid api call from Overpass, but will also increase the cost.
            Defaults to 1. Set to 3 if the message is the first one, in the future
            this could be changed to run whenever the overpass_query function
            is called.

        Returns:
            _type_: _description_
        """

        # This breaks if the messages are not valid
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=self.messages,
            functions=self.function_metadata,
            function_call="auto",
            n=n,
        )
        response_messages = [choice["message"] for choice in response["choices"]]

        # Filter out invalid messages based on your condition
        valid_response_messages = [
            msg for msg in response_messages if self.is_valid_message(msg)
        ]
        invalid_response_messages = [
            "invalid args"
            for msg in response_messages
            if not self.is_valid_message(msg)
        ]

        return valid_response_messages, invalid_response_messages

    def read_plan(self, plan):
        lines = plan.split("\n")
        lines = [l for l in lines if l != ""][1:-1]
        return lines

    def run_conversation(self):
        """Run this after every user message
        originally had the following structure:
        - create overpass query from user message
        - do a function call and return the query result
        - interpret the query result for the user.

        To reduce failures due to hallucinated query formats, three queries are generated in
        response to the first message and are run sequentially until one of them works.

        As user messages get more complex, this may need to be broken down into smaller steps.
        In its current form, the calls are built into a loop which feed each response from the
        large language model back into the conversation history, so that the next response has
        all the previous responses as context.

        # Removing this as it was a hack
        # Increase n to 3 if this is the first user message, to increase chances of a working query
        user_messages = [m for m in self.messages if m["role"] == "user"]
        num_user_messages = len(user_messages)
        if num_user_messages == 1:
            n = 1  # increase the number of initial responses to increase chances of success
            # the chat id is {timestamp}_{first question}
            self.id = self.id + "_" + user_messages[0]["content"]
        else:
            n = 1

        ### UNDER CONSTRUCTION ###
        Change the code below to enable the language model
        to act as an agent and repeat actions until the goal is achieved
        """
        self.latest_question = [
            m["content"] for m in self.messages if m["role"] == "user"
        ][-1]
        print(f"Latest question:{self.latest_question}")

        counter = 0
        while counter < 5:
            # Process messages
            response_messages, invalid_messages = self.process_messages(1)
            self.messages += response_messages
            self.invalid_messages += invalid_messages
            self.plan = []
            self.current_step = 1

            # Check if response includes a function call, and if yes, run it.
            for response_message in response_messages:
                # Check for a plan (should only happen in the first response)
                if (response_message.get("content")) and (
                    response_message.get("content").startswith("Plan:")
                ):
                    self.plan = self.read_plan(response_message.get("content"))
                    print(f"plan: {self.plan}")

                # Check progress
                if (response_message.get("content")) and (
                    "step" in str.lower(response_message.get("content"))
                ):
                    try:
                        self.current_step = int(
                            response_message.get("content").split("Step ")[1][0]
                        )
                    except:
                        print(response_message.get("content"))
                        self.current_step += 1

                if response_message.get("function_call"):
                    if self.function_status_pass:
                        continue  # skips running the next API calls
                    self.execute_function(response_message)

            # if self.is_goal_achieved(response_message):
            #    break
            counter += 1

            filename = f"{self.id} | {self.latest_question}"
            filepath = os.path.join(self.log_path, filename)

            self.save_to_json(
                file_path=filepath,
                this_run_name=f"iteration {counter} step {self.current_step}",
                log={
                    "valid_messages": self.messages,
                    "invalid_messages": self.invalid_messages,
                    "overpass_queries": self.overpass_queries,
                },
            )

        return response_message


# chatbot = ChatBot()
# chatbot.add_user_message(
#     "are there any ping pong tables in Monbijoupark? which one is closest to a toilet?"
# )

# print(chatbot.run_conversation())
