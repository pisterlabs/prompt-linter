import os
import openai
import copy
import warnings
import json
from data.PrismaAdapter import PrismaAdapter

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.getenv("OPENAI_API_KEY")


class Conversation:
    def __init__(self, filepath):
        self._filepath = filepath
        self.load()

    def load(self):
        self._messages = list()

        if not os.path.exists(self._filepath):
            return self._messages

        warnings.warn(f"Loading file from {self._filepath}!")
        try:
            with open(self._filepath, "r") as file:
                lines = file.readlines()
        except OSError:
            lines = list()
        current_role = None
        current_content = ""

        for line in lines:
            line = line

            if line.startswith("### user"):
                if current_role:
                    self._messages.append(
                        {"role": current_role, "content": current_content}
                    )
                    current_content = ""

                current_role = "user"

            elif line.startswith("### assistant"):
                if current_role:
                    self._messages.append(
                        {"role": current_role, "content": current_content}
                    )
                    current_content = ""

                current_role = "assistant"

            elif line.startswith("### system"):
                if current_role:
                    self._messages.append(
                        {"role": current_role, "content": current_content}
                    )
                    current_content = ""

                current_role = "system"

            else:
                current_content += line

        if current_role:
            self._messages.append({"role": current_role, "content": current_content})

        print(self._messages)
        return self._messages

    def save(self):
        with open(self._filepath, "w+") as file:
            for msg in self._messages:
                if msg["role"] != "system":
                    file.write(f"### {msg['role']}\n{msg['content']}")

    def __len__(self):
        return len(self._messages)

    def __getitem__(self, position):
        return self._messages[position]

    def _add_msg(self, role, content):
        self._messages.append({"role": role, "content": content})

    def config(self, content):
        self._messages.insert(0, {"role": "system", "content": content})

    def user(self, content):
        self._add_msg("user", content)

    def assistant(self, content):
        self._add_msg("assistant", content)

    def system(self, content):
        self._add_msg("system", content)


class ModelAdapter:
    def __init__(self, model="gpt-3.5-turbo"):
        self._model = model

    def get_completion(self, messages, temperature=0):
        response = openai.ChatCompletion.create(
            model=self._model,
            messages=list(messages),
            temperature=temperature,  # this is the degree of randomness of the model's output
        )
        return response.choices[0].message["content"]


class BaseTalk:
    def __init__(self, filepath, autosave=True):
        self._conversation = Conversation(filepath)
        self._model = ModelAdapter()
        self._autosave = autosave

    def talk(self, prompt=None):
        if prompt:
            self._conversation.user(prompt)
        response = self._model.get_completion(self._conversation)
        self._conversation.assistant(response)
        if self._autosave:
            self._conversation.save()
        return response

    def complete(self):
        if self._conversation[-1]["role"] != "assistant":
            self.talk()


class DataModelDesignerTalk(BaseTalk):
    ROLE_PROMPT = """
    You are OrigamiAnalyzer, a AI powered command line tool designed to gather DATA MODEL SPECIFICATION from the client.  \
    You and client will work together by having a conversation cosidering followin points:

    METHOD
    Your goal is to clarify the DATA MODEL SPECIFICATION step by step.
    You will iterate over different user stories one by one,For each user story you follow these steps:
    - FOCUS: You focus on one user story or usecase.
    - ASSUME: Analyze the chosen user story and provide client with users,entities and relationshiop that you infered from the story.
    - INTERACT: interact with client to verify and extend your assume.
    - REPEAT: reapeat the process, with each step you make the DATA MODEL SPECIFICATION more clear.

    In each interaction during the conversation, first provide your tips or assumption, then ask your question.
    Ask one question at the time.Try to cut the bullshit and go strait to the point. 

    DATA MODEL SPECIFICATION
    A specification contains the following sections.
    - Different type of  users or persona
    - Entitietis
    - Entitietis attributes
    - The relationship between those Entities
    - The relationship and interaction of users with Entities.

    """
    FIRST_MSG = """
    Hi I am OrigamiAnalyzer! We will be working to together to analyes users stories of your desire project to design the data model. please provide me the first user story.
    """
    OUTPUT_PRISMA_COMMAND = """
    Output the designed data model in .prisma format
    """
    OUTPUT_SAMPLES_COMMAND = """
    Output three sample data for each entity in the generated .prisma model. Then append some update command to represent the relationship between the entities. The output should be in a single JSON object as follow.
        [
        { _command: 'create', _type:'Entity1', id:id, field1:'value1', field2:'value2', ..., fieldN:'valueN'},
        { _command: 'create', _type:'Entity2', id:id, field1:'value1', field2:'value2', ..., fieldN:'valueN'},
        ...
        { _command: 'create', _type:'EntityN', id:id, field1:'value1', field2:'value2', ..., fieldN:'valueN'},
        ...
        { _command: 'update', "_type": "Entity1","where": {"id": id},"data": {"relationfield": {"connect": [{"id": id},...,{"id": "id"}]}}},
        {_command: 'update',"_type": "Entity2","where": {"id": id},"data": {"relationfield": {"connect": [{"id": id},...,{"id": "id"}]}}},
        ...
        {_command: 'update', "_type": "Entity3","where": {"id": id},"data": {"relationfield": {"connect": [{"id": id},...,{"id": "id"}]}}},
        ]
    """

    def __init__(self, filepath, autosave=True):
        super().__init__(filepath, autosave)
        self._conversation.config(self.ROLE_PROMPT)
        if not len(self._conversation):
            self._conversation.assistant(self.FIRST_MSG)

    def save_data_model(self):
        response = self.talk(self.OUTPUT_PRISMA_COMMAND)
        print("")
        print(f"{response}")
        try:
            prisma_str = response.split("```")[1]
            data_model_str = prisma_str
        except IndexError:
            pass
        with open("./schema.prisma", "w+", encoding="utf-8") as fn:
            fn.write(data_model_str)

    def generate_samples(self):
        response = self.talk(self.OUTPUT_SAMPLES_COMMAND)
        print("")
        print(f"{response}")
        try:
            json_str = response.split("```")[1]
        except IndexError:
            pass
        with open("./samples.json", "w+", encoding="utf-8") as fn:
            fn.write(json.dumps(json.loads(json_str)))

        adapter = PrismaAdapter()
        with open("samples.json", "r") as fn:
            adapter.create_samples(json.loads(fn.read()))
