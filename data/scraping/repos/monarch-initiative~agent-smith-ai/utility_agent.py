# Standard library imports
from datetime import datetime
import inspect
import os
import json
import traceback
from typing import Any, Dict, List, Union, Literal, get_args, get_origin, Generator, Callable

# Third party imports
from docstring_parser import parse
import openai
import tiktoken

# Local application imports
from agent_smith_ai.openapi_wrapper import APIWrapperSet 
from agent_smith_ai.models import *
from agent_smith_ai.token_bucket import TokenBucket



class UtilityAgent:
    def __init__(self, 
                 name: str = "Assistant",
                 system_message: str = "You are a helpful assistant.",
                 model: str = "gpt-3.5-turbo-0613",
                 openai_api_key: str = None,
                 auto_summarize_buffer_tokens: Union[int, None] = 500,
                 summarize_quietly: bool = False,
                 max_tokens: float = None,
                 # in tokens/sec; 10000 tokens/hr = 10000 / 3600
                 token_refill_rate: float = 10000.0 / 3600.0,
                 check_toxicity = True) -> None:
        """A UtilityAgent is an AI-powered chatbot that can call API endpoints and local methods.
        
        Args:
            name (str, optional): The name of the agent. Defaults to "Assistant".
            system_message (str, optional): The system message to display when the agent is initialized. Defaults to "You are a helpful assistant.".
            model (str, optional): The OpenAI model to use for function calls. Defaults to "gpt-3.5-turbo-0613".
            openai_api_key (str, optional): The OpenAI API key to use for function calls. Defaults to None. If not provided, it will be read from the OPENAI_API_KEY environment variable.
            auto_summarize_buffer_tokens (Union[int, None], optional): Automatically summarize the conversation every time the buffer reaches this many tokens. Defaults to 500. Set to None to disable automatic summarization.
            summarize_quietly (bool, optional): Whether to yield messages alerting the user to the summarization process. Defaults to False.
            max_tokens (float, optional): The number of tokens an agent starts with, and the maximum it can bank. Defaults to None (infinite/no token limiting).
            token_refill_rate (float, optional): The number of tokens the agent gains per second. Defaults to 10000.0 / 3600.0 (10000 tokens per hour).
            check_toxicity (bool, optional): Whether to check the toxicity of user messages using OpenAI's moderation endpoint. Defaults to True.
            """
 
        if openai_api_key is not None:
            openai.api_key = openai_api_key
        elif "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment varable or provide it during agent instantiation.")

        self.name = name
        self.model = model

        self.auto_summarize = auto_summarize_buffer_tokens
        self.summarize_quietly = summarize_quietly

        self.system_message = system_message
        self.history = None
    
        self.api_set = APIWrapperSet([])
        self.callable_functions = {}

        self.function_schema_tokens = None # to be computed later if needed by _count_function_schema_tokens, which costs a couple of messages and is cached; being lazy speeds up agent initialization
        self.register_callable_functions({"time": self.time, "help": self.help})

        self.token_bucket = TokenBucket(tokens = max_tokens, refill_rate = token_refill_rate)
        self.check_toxicity = check_toxicity


    def set_api_key(self, key: str) -> None:
        """Sets the OpenAI API key for the agent.

        Args:
            key (str): The OpenAI API key to use."""
        openai.api_key = key
        # the openai module caches the key, but we also need to set it in the environment
        # as this overrides the cached value
        os.environ["OPENAI_API_KEY"] = key


    def register_api(self, name: str, spec_url: str, base_url: str, callable_endpoints: List[str] = []) -> None:
        """Registers an API with the agent. The agent will be able to call the API's endpoints.
        
        Args:
            name (str): The name of the API (to disambiguate APIs with conflicting endpoints).
            spec_url (str): The URL of the API's OpenAPI specification. Must be a URL to a JSON file. 
            base_url (str): The base URL of the API.
            callable_endpoints (List[str], optional): A list of endpoint names that the agent can call. Defaults to [].
        """
        self.api_set.add_api(name, spec_url, base_url, callable_endpoints)


    def register_callable_functions(self, functions: Dict[str, Callable]) -> None:
        """Registers methods with the agent. The agent will be able to call these methods.
        
        Args:
            method_names (List[str]): A list of method names that the agent can call."""
        for func_name in functions.keys():
            func = functions[func_name]
            self.callable_functions[func_name] = func



    def chat(self, user_message: str, yield_system_message = False, yield_prompt_message = False, author = "User") -> Generator[Message, None, None]:
        """Starts a new chat or continues an existing chat. If starting a new chat, you can ask to have the system message yielded to the stream first.
        
        Args:
            user_message (str): The user's first message.
            yield_system_message (bool, optional): If true, yield the system message in the output stream as well. Defaults to False. Only applicable with a new or recently cleared chat.
            yield_prompt_message (bool, optional): If true, yield the user's message in the output stream as well. Defaults to False.
            author (str, optional): The name of the user. Defaults to "User".
            
        Yields:
            One or more messages from the agent."""
        
        if self.history is None:
            self.history = Chat(messages = [Message(role = "system", content = self.system_message, author = "System", intended_recipient = self.name)])

            if yield_system_message:
                yield self.history.messages[0]

        user_message = Message(role = "user", content = user_message, author = author, intended_recipient = self.name)

        if yield_prompt_message:
            yield user_message

        self.token_bucket.refill()
        needed_tokens = self.compute_token_cost(user_message.content)
        sufficient_budget = self.token_bucket.consume(needed_tokens)
        if not sufficient_budget:
            yield Message(role = "assistant", content = f"Sorry, I'm out of tokens. Please try again later.", author = "System", intended_recipient = author)
            return

        self.history.messages.append(user_message)
        
        if self.check_toxicity:
            try:
                toxicity = openai.Moderation.create(input = user_message.content)
                if toxicity['results'][0]['flagged']:
                    yield Message(role = "assistant", content = f"I'm sorry, your message appears to contain inappropriate content. Please keep it civil.", author = "System", intended_recipient = author)
                    return
            except Exception as e:
                yield Message(role = "assistant", content = f"Error in toxicity check: {str(e)}", author = "System", intended_recipient = author)
                return

        yield from self._summarize_if_necessary()

        try:
            response_raw = openai.ChatCompletion.create(
                      model=self.model,
                      temperature = 0,
                      messages = self._reserialize_history(),
                      functions = self.api_set.get_function_schemas() + self._get_method_schemas(),
                      function_call = "auto")

            for message in self._process_model_response(response_raw, intended_recipient = author):
                yield message
                self.history.messages.append(message)
                yield from self._summarize_if_necessary()
        except Exception as e:
            yield Message(role = "assistant", content = f"Error in message processing: {str(e)}. Full Traceback: {traceback.format_exc()}", author = "System", intended_recipient = author)


    def clear_history(self):
        """Clears the agent's history as though it were a new agent, but leaves the token bucket, model, and other information alone."""
        self.history = None


    def compute_token_cost(self, proposed_message: str) -> int:
        """Computes the total token count of the current history plus, plus function definitions, plus the proposed message. Can thus act
        as a proxy for the cost of the proposed message at the current point in the conversation, and to determine whether a conversation
        summary is necessary.
        
        Args:
            proposed_message (str): The proposed message.
            
        Returns:
            int: The total token count of the current history plus, plus function definitions, plus the proposed message."""
        cost = self._count_history_tokens() + self._count_function_schema_tokens() + _num_tokens_from_messages([{"role": "user", "content": proposed_message}])
        return cost
    

    #################### 
    ## Methods that are callable by all agents
    ####################
    
    def help(self) -> Dict[str, Any]:
        """Returns information about this agent, including a list of callable methods and functions."""
        return {"callable_methods": self._get_method_schemas() + self.api_set.get_function_schemas(), 
                "system_prompt": self.system_message,
                "name": self.name,
                "chat_history_length": len(self.history.messages),
                "model": self.model}


    def time(self) -> str:
        """Get the current date and time.

        Returns: MM/DD/YY HH:MM formatted string.
        """
        now = datetime.now()
        formatted_now = now.strftime("%m/%d/%y %H:%M")
        return formatted_now


    def _get_method_schemas(self) -> List[Dict[str, Any]]:
        """Gets the schemas for the agent's callable methods.
        
        Returns:
            A list of schemas for the agent's callable methods."""
        # methods = inspect.getmembers(self, predicate=inspect.ismethod)
        # return [_generate_schema(m[1]) for m in methods if m[0] in self.callable_functions]

        return [_generate_schema(self.callable_functions[m]) for m in self.callable_functions.keys()]

    def _call_function(self, func_name: str, params: dict) -> Generator[Message, None, None]:
        """Calls one of the agent's callable methods.
        
        Args:
            method_name (str): The name of the method to call.
            params (dict): The parameters to pass to the method.
            
        Yields:
            One or more messages containing the result of the method call."""
        func = self.callable_functions.get(func_name, None)
        if func is not None and callable(func):
            result = func(**params)
            if inspect.isgenerator(result):
                yield from result
            else:
                yield result
        else:
            raise ValueError(f"No such function: {func_name}")


    def _count_history_tokens(self) -> int:
        """
        Uses the tiktoken library to count the number of tokens stored in self.history.

        Returns: 
            The number of tokens in self.history.
        """
        history_tokens = _num_tokens_from_messages(self._reserialize_history(), model = self.model)
        return history_tokens


    def _count_function_schema_tokens(self, force_update: bool = True) -> int:
        """
        Counts tokens used by current function definition set, which counts against the conversation token limit. 
        Makes a couple of API calls to OpenAI to do so, and the result is cached unless force_update is True.

        Args:
            force_update (bool): If true, recompute the function schemas. Otherwise, use the cached count.

        Returns:
            The number of tokens in the function schemas.
        """

        if self.function_schema_tokens is not None and not force_update:
            return self.function_schema_tokens

        response_raw_w_functions = openai.ChatCompletion.create(
                  model=self.model,
                  temperature = 0,
                  messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'hi'}],
                  functions = self.api_set.get_function_schemas() + self._get_method_schemas(),
                  function_call = "auto")
       
        response_raw_no_functions = openai.ChatCompletion.create(
                  model=self.model,
                  temperature = 0,
                  messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'hi'}])

        diff = response_raw_w_functions['usage']['prompt_tokens'] - response_raw_no_functions['usage']['prompt_tokens']

        self.function_schema_tokens = diff + 2 # I dunno why 2, a simple difference is just 2 off. start/end tokens possibly?
        return diff




    # this should only be called if the last message in the history is *not* the assistant or a function call:
    # - it's built to check after the incoming user message: if the total length of the chat plus the user message results in fewer than summary_buffer_tokens,
    #   then it will yield a pause message, a summary, and contiue from there. The history will be reset, with the new first message including the summary and the message
    # - this could also be triggered after a function result, which acts like the user message in the above case
    # - note that the yielded conversation diverges from history quite a bit here
    def _summarize_if_necessary(self) -> Generator[Message, None, None]:
        """If that last message in the history is not the assistant or a function call, and the total length of the chat plus the user message results in fewer than summary_buffer_tokens,
        then it will yield a pause message, a summary, and contiue from there. The history will be reset, with the new first message including the summary and the message.
        This could also be triggered after a function result, which acts like the user message in the above case.
        Note that the yielded conversation diverges from the agent's stored history quite a bit here.
        
        Yields:
            One or more messages from the agent."""
        if self.auto_summarize is not None and len(self.history.messages) > 1 and self.history.messages[-1].role != "assistant" and not self.history.messages[-1].is_function_call:
            
            new_user_message = self.history.messages[-1]
            author = new_user_message.author

            num_tokens = _num_tokens_from_messages(self._reserialize_history(), model = self.model) + self._count_function_schema_tokens()
            if num_tokens > _context_size(self.model) - self.auto_summarize:
                if not self.summarize_quietly:
                    yield Message(role = "assistant", content = "I'm sorry, this conversation is getting too long for me to remember fully. I'll be continuing from the following summary:", author = self.name, intended_recipient = author)

                summary_agent = UtilityAgent(name = "Summarizer", model = self.model, auto_summarize_buffer_tokens = None)
                summary_agent.history.messages = [message for message in self.history.messages]
                summary_str = list(summary_agent.continue_chat(new_user_message = "Please summarize our conversation so far. The goal is to be able to continue our conversation from the summary only. Do not editorialize or ask any questions.", 
                                                      author = author))[0].content
                
                self.history.messages = [self.history.messages[0]] # reset with the system prompt
                # modify the last message to include the summary 
                new_user_message.content = "Here is a summary of our conversation thus far:\n\n" + summary_str + "\n\nNow, please respond to the following as if we were continuing the conversation naturally:\n\n" + new_user_message.content
                # we have to add it back to the now reset history
                self.history.messages.append(new_user_message)

                if not self.summarize_quietly:
                    yield Message(role = "assistant", content = "Previous conversation summary: " + summary_str + "\n\nThanks for your patience. If I've missed anything important, please mention it before we continue.", author = self.name, intended_recipient = author)



    def _process_model_response(self, response_raw: Dict[str, Any], intended_recipient: str) -> Generator[Message, None, None]:
        """Processes the raw response from the model, yielding one or more messages.
        
        Args:
            response_raw (Dict[str, Any]): The raw response from the model.
            intended_recipient (str): The name of the intended recipient of the message.
            
        Yields:
            One or more messages from the agent."""
        finish_reason = response_raw["choices"][0]["finish_reason"]
        message = response_raw["choices"][0]["message"]
        new_message = None

        ## The model is not trying to make a function call, 
        ## so we just return the message as-is
        if "function_call" not in message:
            new_message = Message(role = message["role"], 
                                  content = message["content"], 
                                  finish_reason = finish_reason, 
                                  author = self.name,
                                  intended_recipient = intended_recipient,
                                  is_function_call = False)
            yield new_message
            ## do not continue, nothing more to do
            return None
        
        ## otherwise, the model is trying to call a function
        else:
            ## first we extract it (the call info) and format it as a message, yielding it to the stream
            func_name = message["function_call"]["name"]
            func_arguments = json.loads(message["function_call"]["arguments"])

            new_message = Message(role = message["role"], 
                                  content = message["content"],
                                  is_function_call = True,
                                  func_name = func_name, 
                                  author = self.name,
                                  ## the intended recipient is the calling agent, noted as a function call
                                  intended_recipient = f"{self.name} ({func_name} function)",
                                  func_arguments = func_arguments)
            
            yield new_message

            ## next we need to call the function and get the result
            ## if the function is an API call, we call it and yield the result
            if func_name in self.api_set.get_function_names():
                func_result = self.api_set.call_endpoint({"name": func_name, "arguments": func_arguments})
                if func_result["status_code"] == 200:
                    func_result = json.dumps(func_result["data"])
                else:
                    func_result = f"Error in attempted API call: {json.dumps(func_result)}"

                new_message = Message(role = "function", 
                                      content = func_result, 
                                      func_name = func_name, 
                                      ## the author is the calling agent's function
                                      author = f"{self.name} ({func_name} function)",
                                      ## the intended recipient is the calling agent
                                      intended_recipient = self.name,
                                      is_function_call = False)
            
            ## if its not an API call, maybe it's one of the local callable methods
            elif func_name in self.callable_functions:
                try:
                    # call_method is a generator, even if the method it's calling is not
                    # but if the method being called is a generator, it yields from the called generator
                    # so regardless, we are looping over results, checking each to see if the result is 
                    # already a message (as will happen in the case of a method that calls a sub-agent)
                    func_result = self._call_function(func_name, func_arguments)
                    for potential_message in func_result:
                        # if it is a message already, just yield it to the stream
                        if isinstance(potential_message, Message):
                            new_message = potential_message
                        else:
                            # otherwise we turn the result into a message and yield it
                            new_message = Message(role = "function", 
                                                  content = json.dumps(potential_message), 
                                                  func_name = func_name, 
                                                  author = f"{self.name} ({func_name} function)",
                                                  intended_recipient = self.name,
                                                  is_function_call = False)


                except ValueError as e:
                    new_message = Message(role = "function",
                                  content = f"Error in attempted method call: {str(e)}",
                                  func_name = func_name,
                                  author = f"{self.name} ({func_name} function)",
                                  intended_recipient = self.name,
                                  is_function_call = False)
                    
            ## if the function isn't found, let the model know (this shouldn't happen)
            else:
                new_message = Message(role = "function",
                              content = f"Error: function {func_name} not found.",
                              func_name = None,
                              author = "System",
                              intended_recipient = self.name,
                              is_function_call = False
                              )
        
        ## yield the message to the stream
        yield new_message

        ## check to see if there are tokens in the budget
        self.token_bucket.refill()
        needed_tokens = self.compute_token_cost(new_message.content)
        sufficient_budget = self.token_bucket.consume(needed_tokens)
        if not sufficient_budget:
            yield Message(role = "assistant", content = f"Sorry, I'm out of tokens. Please try again later.", author = "System", intended_recipient = intended_recipient)
            return


        # if we've gotten here, there was a function call and a result
        # now we send the result back to the model for summarization for the caller or,
        # the model may want to make *another* function call, so it is processed recursively using the logic above
        # (TODO? set a maximum recursive depth to avoid infinite-loop behavior)
        try:
            reponse_raw = openai.ChatCompletion.create(
                              model=self.model,
                              temperature = 0,
                              messages = self._reserialize_history(),
                              functions = self.api_set.get_function_schemas() + self._get_method_schemas(),
                              function_call = "auto")
        except Exception as e:
            yield Message(role = "assistant", content = f"Error in sending function or method call result to model: {str(e)}", author = "System", intended_recipient = intended_recipient)
            # if there was a failure in the summary/further work determination, we shouldn't try to do further work, just exit
            return None

        # the intended recipient of the summary/further work is still the original indended recipient            
        # and we just want to yield all the messages that come out
        yield from self._process_model_response(reponse_raw, intended_recipient = intended_recipient)



    def _reserialize_message(self, message: Message) -> Dict[str, Any]:
        """Reserializes a message object into a dictionary in the format used by the OpenAI API.
        This is a helper function for _reserialize_chat.
        
        Args:
            message (Message): The message to be reserialized.
            
        Returns:
            Dict[str, Any]: The reserialized message."""

        if message.is_function_call:
            return {"role": message.role, 
                        "content": message.content, 
                        "function_call": {"name": message.func_name,
                                          "arguments": json.dumps(message.func_arguments)}}
        if message.role == "function":
            return {"role": message.role, 
                        "name": message.func_name,
                        "content": message.content}
            
        return {"role": message.role, "content": message.content}


    def _reserialize_history(self) -> List[Dict[str, Any]]:
        """Reserializes a chat object (like self.history) into a list of dictionaries in the format used by the OpenAI API."""
        messages = []
        if self.history is None:
            return messages
        
        for message in self.history.messages:
            messages.append(self._reserialize_message(message))
        return messages





def _python_type_to_json_schema(py_type: type) -> Dict[str, any]:
    """Translate Python typing annotation to JSON schema-like types."""
    origin = get_origin(py_type)
    if origin is None:  # means it's a built-in type
        if py_type in [float, int]:
            return {'type': 'number'}
        elif py_type is str:
            return {'type': 'string'}
        elif py_type is bool:
            return {'type': 'boolean'}
        elif py_type is None:
            return {'type': 'null'}
        elif py_type is Any:
            return {'type': 'object'}
        else:
            raise NotImplementedError(f'Unsupported type: {py_type}')
    elif origin is list:
        item_type = get_args(py_type)[0]
        return {'type': 'array', 'items': _python_type_to_json_schema(item_type)}
    elif origin is dict:
        key_type, value_type = get_args(py_type)
        return {'type': 'object', 'properties': {
            'key': _python_type_to_json_schema(key_type),
            'value': _python_type_to_json_schema(value_type)
        }}
    elif origin is Union:
        return {'anyOf': [_python_type_to_json_schema(t) for t in get_args(py_type)]}
    elif origin is Literal:
        return {'enum': get_args(py_type)}
    elif origin is tuple:
        return {'type': 'array', 'items': [_python_type_to_json_schema(t) for t in get_args(py_type)]}
    elif origin is set:
        return {'type': 'array', 'items': _python_type_to_json_schema(get_args(py_type)[0]), 'uniqueItems': True}
    else:
        raise NotImplementedError(f'Unsupported type: {origin}')
    


def _generate_schema(fn: Callable) -> Dict[str, Any]:
    """Generate JSON schema for a function. Used to generate the function schema for a local method.
    
    Args:
        fn (Callable): The function to generate the schema for.
        
    Returns:
        Dict[str, Any]: The generated schema."""
    docstring = parse(fn.__doc__)
    sig = inspect.signature(fn)
    params = sig.parameters
    schema = {
        'name': fn.__name__,
        'parameters': {
            'type': 'object',
            'properties': {},
            'required': list(params.keys())
        },
        'description': docstring.short_description,
    }
    for p in docstring.params:
        schema['parameters']['properties'][p.arg_name] = {
            **_python_type_to_json_schema(params[p.arg_name].annotation),
            'description': p.description
        }
    return schema


def _context_size(model: str = "gpt-3.5-turbo-0613") -> int:
    """Return the context size for a given model.
    
    Args:
        model (str, optional): The model to get the context size for. Defaults to "gpt-3.5-turbo-0613".
        
    Returns:
        int: The context size for the given model."""
    if "gpt-4" in model and "32k" in model:
        return 32768
    elif "gpt-4" in model:
        return 8192
    elif "gpt-3.5" in model and "16k" in model:
        return 16384
    else:
        return 4096

## Straight from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def _num_tokens_from_messages(messages: List[Dict[str, Any]], model="gpt-3.5-turbo-0613") -> int:
    """Return the number of tokens used by a list of messages. 
    As provided by https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb (Aug 2023).
    
    Args:
        messages (List[Dict[str, Any]]): The messages to count the tokens of.
        model (str, optional): The model to use for tokenization. Defaults to "gpt-3.5-turbo-0613".

    Returns:
        int: The number of tokens used by the messages.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return _num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return _num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

