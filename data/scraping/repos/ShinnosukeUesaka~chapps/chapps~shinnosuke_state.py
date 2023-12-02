"""Base state for the app."""
import reflex as rx
from chapps import firebase_utils
import yaml
from openai_utils import create_chat_and_parse, GPTConfig, create_chat
from firebase_utils import db
import uuid

class User(rx.Base):
    id: str
    name: str


class Example(rx.Base):
    inputs: dict[str, str]
    output: str


class Chapp(rx.Base):
    id: str
    user: str
    title: str
    short_description: str
    description: str
    icon_url: str = None
    inputs: list[str]
    examples: list[Example]
    instruction: str


# class State(rx.State):
class State:
    """Base state for the app.

    The base state is used to store general vars used throughout the app.
    """

    user: User = None
    logged_in: bool = False

    def create_user_or_login(self, name):
        user_data = firebase_utils.add_or_get_user(name)
        self.user.id = user_data["id"]
        self.user.name = user_data["name"]

        self.logged_in = True


class HomeState(State):
    my_chapps: list[Chapp]

    def get_chaps(self):
        data = db.collection("chapps").where("user", "==", self.user.id).stream()
        for doc in data:
            chapp = doc.to_dict()
            self.my_chapps.append(Chapp(**chapp))

class RunChappState(State):
    chapp: Chapp = None
    inputs: list[str]
    output: str
    status: str

    def run_chapp(self):
        self.output = call_chap(self.chapp, self.inputs)

class CreateNewState(State):
    description_of_chapp: str = None
    chapp: Chapp = None

    def create_chapp(self):
        chapp = create_chapp(self.description_of_chapp, self.user.id)
        self.chapp = chapp

    def edit_title(self, text):
        self.chapp.title = text

    def edit_short_description(self, text):
        self.chapp.short_description = text

    def edit_icon_url(self, text):
        self.chapp.icon_url = text

    def edit_intruction(self, text):
        self.chapp.instruction = text

    def edit_inputs(self):
        #TODO
        pass

    def edit_examples(self):
        #TODO
        pass

    def save_chapp(self):
        db.collection("chapps").document(self.chapp.id).set(self.chapp.dict())

class ExploreState(State):
    search_query: str = None
    search_results: list[Chapp] = []

    def search_chaps(self, serach_query):
        query = self.search_query
        #TODO semantic search
        # for now random search with limit of 5.
        data = db.collection("chapps").limit(5).stream()
        for doc in data:
            chapp = doc.to_dict()
            self.search_results.append(Chapp(**chapp))


def create_chapp(description: str, user_id: str) -> Chapp:
    prompt = """
Your job is to create a Chapp based on the prompt below. Chapp is a tool or an app that runs on GPT-4.

Give me the all the variables neccesary to define the chapp.
A chapp has a title, description, short description inputs variables(all lowercase and use underscore for multiple words), instruction(prompt for gpt-4), example pair of inputs and outputs. All the input variables are string.
The output must strictly follow the example yaml format below, as it would be parsed programatically.


Example Input
I want a tool that gives me a definition of a word, and three example sentences based on a context provided.

Example Output

title: Word Context Definition and Example Builder

short_description: A tool for learning new words and their usage in a specific context.

description: |-
   This Chapp provides you the definition of a specific word and constructs three sentences using that word, based around a specific context provided by the user. It's a tool that can be handy for learning new words, enhancing your vocabulary, and understanding the usage of a word in a context effectively.

inputs:
  - id: word
    description: Enter the word you want to learn about and see used in sentences.
  - id: context
    description: Specify the context or theme within which you want to see the word used.
instruction: |-
  For the word "{word}", first provide a clear and concise definition. Then, based on the context of "{context}", create three unique sentences that correctly use and demonstrate the meaning of the word.

example:
  input:
    word: procrastinate
    context: school
  output: |-
    ## Definition
    Procrastinate means to delay or postpone action; put off doing something.
    ## Example Sentences:
    1. Many students tend to procrastinate when it comes to studying for exams, often leading to stress and poor performance.
    2. In school, procrastinating on assignments can result in late submissions and penalties.
    3. Despite knowing the importance of timely work, John often found himself procrastinating on his school projects.
"""

    def parsing_function(yaml_str: str) -> Chapp:
        chapp_data = yaml.safe_load(yaml_str)
        uid = uuid.uuid4().hex
        chapp["id"] = uid
        chapp["user"] = user_id
        return Chapp(**chapp_data)

    messages = [
        {"role": "user", "content": prompt},
        {"role": "user", "content": description},
    ]
    chapp = create_chat_and_parse(
        messages=messages, parsing_function=parsing_function, gpt_config=GPTConfig()
    )

    return chapp


def call_chap(chapp: Chapp, inputs: dict[str, str]) -> str:
    formated_instruction = chapp.instruction.format(**inputs)
    example_string = "Example Input\n"
    for example in chapp.examples:
        for key, value in example.inputs.items():
            example_string += f"{key}: {value}\n"
        example_string += f"\nExample Output\n{example.output}"


    prompt = f"""{formated_instruction}
Strictly follow the format of the example below.

{example_string}
"""

    return create_chat(messages=[{"role": "user", "content": prompt}], gpt_config=GPTConfig())
