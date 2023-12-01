"""Creates the Example and GPT classes for a user to interface with the OpenAI
API."""

import openai
import uuid

from openai.api_requestor import APIRequestor

from data import prompt_text


def set_openai_key(key, api_version=None, api_base='https://api.openai.com/v1', api_type='open_ai'):
    """Sets OpenAI key."""
    openai.api_key = key
    openai.api_version = api_version
    openai.api_type = api_type
    openai.api_base = api_base


class Example:
    """Stores an input, output pair and formats it to prime the model."""

    def __init__(self, inp, out):
        self.input = inp
        self.output = out
        self.id = uuid.uuid4().hex

    def get_input(self):
        """Returns the input of the example."""
        return self.input

    def get_output(self):
        """Returns the intended output of the example."""
        return self.output

    def get_id(self):
        """Returns the unique ID of the example."""
        return self.id

    def as_dict(self):
        return {
            "input": self.get_input(),
            "output": self.get_output(),
            "id": self.get_id()
        }

    def __str__(self):
        return f'Q:{self.input}\nA:{self.output}'


class GPT:
    """The main class for a user to interface with the OpenAI API.
    A user can add examples and set parameters of the API request.
    """

    def __init__(self,
                 engine='davinci',
                 temperature=0.5,
                 max_tokens=100,
                 input_prefix="输入: ",
                 input_suffix="\n",
                 output_prefix="",
                 output_suffix="\n\n",
                 append_output_prefix_to_query=False):
        self.examples = {}
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.input_prefix = input_prefix
        self.input_suffix = input_suffix
        self.output_prefix = output_prefix
        self.output_suffix = output_suffix
        self.append_output_prefix_to_query = append_output_prefix_to_query
        self.stop = (output_suffix + input_prefix).strip()
        self.role = 'user'

    def add_example(self, ex):
        """Adds an example to the object.
        Example must be an instance of the Example class.
        """
        assert isinstance(ex, Example), "Please create an Example object."
        self.examples[ex.get_id()] = ex
        return ex

    def delete_example(self, id):
        """Delete example with the specific id."""
        if id in self.examples:
            del self.examples[id]

    def get_example(self, id):
        """Get a single example."""
        return self.examples.get(id, None)

    def get_all_examples(self):
        """Returns all examples as a list of dicts."""
        return {k: v.as_dict() for k, v in self.examples.items()}

    def get_prime_text(self):
        """Formats all examples to prime the model."""
        return "".join(
            [self.format_example(ex) for ex in self.examples.values()])

    def get_engine(self):
        """Returns the engine specified for the API."""
        return self.engine

    def get_temperature(self):
        """Returns the temperature specified for the API."""
        return self.temperature

    def get_max_tokens(self):
        """Returns the max tokens specified for the API."""
        return self.max_tokens

    def craft_query(self, prompt):
        """Creates the query for the API request."""
        q = self.get_prime_text() + self.input_prefix + prompt + self.input_suffix
        if self.append_output_prefix_to_query:
            q = q + self.output_prefix
        return q

    def get_query_message(self, content, context):
        """Create the query for the API request."""
        user_group = [{"role": self.role, "content": content}]
        if context:
            sys_user = {"role": "system",
                        "content": f'''您是一个AI助手，您得到了以下长文档的提取部分和一个问题。根据提供的上下文等进行交流式回答。"
                                   f"基于以下内容，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 "不知道" 或 "没有足够的相关信息"，不要试图编造答案。答案请使用中文。"
                                   f"然后请说出自行回答的内容“我的个人分析是：\n{context}'''
                        }
            user_group.append(sys_user)
        return user_group

    def generate_prompt(self, text, task_type):
        """
        Generate prompt info
        """
        prompt = f"{prompt_text[task_type]} {text.strip()}"
        if self.get_prime_text():
            q = self.get_prime_text() + self.input_prefix + prompt + self.input_suffix
        else:
            q = prompt + self.output_suffix
        return q

    def submit_request(self, text, task_type, context, model_type):
        """Calls the OpenAI API with the specified parameters.
        """
        response = openai.ChatCompletion.create(engine=self.get_engine() if model_type == 'azure' else None,
                                                model=self.get_engine() if model_type == 'open_ai' else None,
                                                messages=self.get_query_message(self.generate_prompt(text, task_type),
                                                                                context=context),
                                                max_tokens=self.get_max_tokens(),
                                                temperature=self.get_temperature()
                                                )
        return response

    def get_top_reply(self, text, task_type, context='', model_type=''):
        """Obtains the best result as returned by the API."""
        response = self.submit_request(text, task_type, context, model_type)
        return response.choices[0]['message']['content'], response['usage']['total_tokens']

    def format_example(self, ex):
        """Formats the input, output pair."""
        return self.input_prefix + ex.get_input(
        ) + self.input_suffix + self.output_prefix + ex.get_output(
        ) + self.output_suffix
