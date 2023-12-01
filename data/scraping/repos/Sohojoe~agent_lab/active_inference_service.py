import asyncio
import datetime
from dateutil import parser
from enum import Enum
import itertools
import json
import os
import traceback
import openai
from openai import AsyncOpenAI

from pydantic import BaseModel, Field, PrivateAttr
from typing import Any, Callable, ForwardRef, List, Optional, Sequence
from generative_model import GenerativeModel

from langchain_helper import convert_pydantic_to_openai_function, create_instance_from_response
from sensory_stream import SensoryStream

Policy = ForwardRef('Policy')
update_generative_model_fn = ForwardRef('update_generative_model_fn')
track_policy_progress_fn = ForwardRef('track_policy_progress_fn')
select_policy_fn = ForwardRef('select_policy_fn')

class ActiveInferenceService:
    def __init__(self, api="openai", fast_model_id = "gpt-3.5-turbo", best_model_id="gpt-4"):
        self._api = api
        self._aclient = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self._fast_model_id = fast_model_id
        self._best_model_id = best_model_id


    async def invoke_llm_async(self, messages, functions, use_best=False, cancel_event=None):
        delay = 0.1
        openai_functions = [convert_pydantic_to_openai_function(f) for f in functions]
        fn_names = [oai_fn["function"]["name"] for oai_fn in openai_functions]
        # function_call="auto" if len(functions) > 1 else f"{{'name': '{fn_names[0]}'}}"
        function_call="auto" if len(functions) > 1 else {
            "type": "function", 
            "function": {"name": fn_names[0]}
        }
        # function_call="auto"
        model_id = self._best_model_id if use_best else self._fast_model_id

        while True:
            try:
                response = await self._aclient.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=1.0,
                    tools=openai_functions,
                    tool_choice=function_call,
                    stream=False
                )
                output = response.choices[0].message
                function_instances = create_instance_from_response(output, functions)
                if len(function_instances) == 0:
                    raise Exception("No function call in the response.")
                if len(function_instances) > 1:
                    raise Exception("only 1 function call currently supported.")
                return function_instances[0]

            except openai.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
                print(f"Retrying in {delay} seconds...")

            # except openai.error.APIConnectionError as e:
            #     print(f"Failed to connect to OpenAI API: {e}")
            #     print(f"Retrying in {delay} seconds...")

            # except openai.error.RateLimitError as e:
            #     print(f"OpenAI API request exceeded rate limit: {e}")
            #     print(f"Retrying in {delay} seconds...")

            except Exception as e:
                print(f"OpenAI API unknown error: {e}")
                trace = traceback.format_exc()
                print(f"trace: {trace}")
                print(f"Retrying in {delay} seconds...")

            await asyncio.sleep(delay)
            delay *= 2

    async def select_policy(self, sensory_stream: SensoryStream, generative_model: GenerativeModel)->select_policy_fn:
        messages = []
        # before_policy_stream, after_policy_stream = sensory_stream.pritty_print_split(cur_policy._time_stamp)
        # before_policy_stream = before_policy_stream.split("\n")
        # after_policy_stream = after_policy_stream.split("\n")
        stream = sensory_stream.pritty_print().split("\n")
        system_prompt = f"""
You are an Artificial Intelligence expert specializing in Active Inference, the Free Energy Principle, and the Markov Blanket. Your landmark research showed that Large Language Models (LLMs) like GPT-4 can perform Active Inference.

Analyze the following input stream and make any modifications to the generative model (the hidden states, beliefs, desires) from the perspective of the assistant.
*** Input Stream Since Last Model Update***
```json
{stream}
```

*** Current/Previous Generative Model ***
```json
{generative_model.model_dump_json()}
```

"""
        messages.append({"role": "system", "content": system_prompt})
        functions = [
            select_policy_fn
        ]
        select_policies_result: select_policy_fn = await self.invoke_llm_async(messages, functions, use_best=False)
        policy_impacts = [p.estimated_free_energy_reduction * p.probability_of_success for p in select_policies_result.policies]
        max_index = policy_impacts.index(max(policy_impacts))
        if select_policies_result.selected_policy_idx < 0 or select_policies_result.selected_policy_idx > len(select_policies_result.policies):
            select_policies_result.selected_policy_idx = max_index
        return select_policies_result

    def _pretty_print_time_since(self, time_stamp):
        current_utc_time = datetime.datetime.utcnow()
        past_time = parser.parse(time_stamp)
        time_since = (current_utc_time - past_time).total_seconds()
        if time_since < 1:
            return "just now"
        if time_since < 60:
            return f"{int(time_since)} seconds ago"
        elif time_since < 3600:
            return f"{int(time_since / 60)} minutes ago"
        elif time_since < 86400:
            return f"{int(time_since / 3600)} hours ago"
        else:
            return f"{int(time_since / 86400)} days ago"
        
    async def track_policy_progress(self, sensory_stream: SensoryStream, generative_model: GenerativeModel, cur_policy: Policy)->track_policy_progress_fn:
        messages = []
        # stream = sensory_stream.pritty_print().split("\n")
        before_policy_stream, after_policy_stream = sensory_stream.pritty_print_split(cur_policy._time_stamp)
        before_policy_stream = before_policy_stream.split("\n")
        after_policy_stream = after_policy_stream.split("\n")
        policy_age = self._pretty_print_time_since(cur_policy._time_stamp)
        after_policy_stream.append(f"time_since_policy: {policy_age}")

        system_prompt = f"""
You are an Artificial Intelligence expert specializing in Active Inference, the Free Energy Principle, and the Markov Blanket. Your landmark research showed that Large Language Models (LLMs) like GPT-4 can perform Active Inference.

Analyze the input stream:

*** Assistant's Current Policy ***
```json
{cur_policy.model_dump_json()}
```
"""
        messages.append({"role": "system", "content": system_prompt})
        for message in after_policy_stream:
            messages.append({"role": "user", "content": message})
        functions = [
            track_policy_progress_fn
        ]
        updates: track_policy_progress_fn = await self.invoke_llm_async(messages, functions, use_best=False)
        return updates
    
    async def update_generative_model(self, sensory_stream: SensoryStream, generative_model: GenerativeModel, cur_policy: Policy)->update_generative_model_fn:
        messages = []
        # stream = sensory_stream.pritty_print().split("\n")
        before_policy_stream, after_policy_stream = sensory_stream.pritty_print_split(cur_policy._time_stamp)
        before_policy_stream = before_policy_stream.split("\n")
        after_policy_stream = after_policy_stream.split("\n")
        policy_age = self._pretty_print_time_since(cur_policy._time_stamp)
        after_policy_stream.append(f"time_since_policy: {policy_age}")

        system_prompt = f"""
You are an Artificial Intelligence expert specializing in Active Inference, the Free Energy Principle, and the Markov Blanket. Your landmark research showed that Large Language Models (LLMs) like GPT-4 can perform Active Inference.

Analyze the following input stream and make any modifications to the generative model (the hidden states, beliefs, desires) from the perspective of the assistant.

*** Current/Previous Generative Model ***
```json
{generative_model.model_dump_json()}
```

*** Assistant's Current Policy ***
```json
{cur_policy.model_dump_json()}
```
"""
        messages.append({"role": "system", "content": system_prompt})
        for message in after_policy_stream:
            messages.append({"role": "user", "content": message})
        functions = [
            update_generative_model_fn
        ]
        updates: update_generative_model_fn = await self.invoke_llm_async(messages, functions, use_best=False)
        return updates

def _time_stamp():
    utc_time = datetime.datetime.utcnow()
    return utc_time.isoformat()

class Policy(BaseModel):
    """A set of actions the assistant takes to reduce free energy"""
    policy: str = Field(..., description="The set of actions in the policy")
    expected_outcome:str = Field(..., description="expected changes that the policy will produce in perception, hidden_states, and/or beliefs")
    estimated_free_energy_reduction: float = Field(...,  description= "how much free energy the policy will remove from the system. int between 1 and 10")
    probability_of_success: float = Field(...,  description= "how likley the assistant is to succeed with this policy, between 0 and 1")
    _time_stamp: str = PrivateAttr(default_factory=_time_stamp)

# class EditBelifActionEnum(str, Enum):
#     create = "create"
#     edit = "edit"
#     delete = "delete"


class AddBelief(BaseModel):
    """Create a belief"""
    belief: str = Field(..., description="something the assistent beleives about themselves, the user, or the world")

class DeleteBelief(BaseModel):
    """Delete a belief"""
    belief: str = Field(None, description="belief to detele. MUST match an existing belief")


class EditBelief(BaseModel):
    """Edit, create, or delete a belief"""
    new_belief: str = Field(..., description="something the assistent beleives about themselves, the user, or the world")
    old_belief: str = Field(None, description="belief to update. MUST match an existing belief")

class PolicyIsCompleteEnum(str, Enum):
    """Is the policy complete? Should the assistant continue or interrupt?"""
    continue_policy = "continue",
    complete = "complete",
    interrupt_policy = "interrupt",

class update_generative_model_fn(BaseModel):
    """
    """
    generative_model: GenerativeModel = Field(..., description="updated generative model")

class track_policy_progress_fn(BaseModel):
    """ Important: only use information in the Input Stream....

    """
    # policy_progress: str = Field(..., description="follow rules above for policy_progress")
    question_1: str = Field(..., description="what progress has the assistant made on the policy?")
    question_2: str = Field(..., description="has the expected_outcome been achieved?")
    question_3: str = Field(..., description="is it still likley the expected_outcome will be achieved?")
    policy_is_complete: PolicyIsCompleteEnum = Field(..., description="continue_policy, complete, or interrupt_policy")

class FreeEnergy(BaseModel):
    cause: str = Field(..., description="cause of free energy / uncertanty in the system")
    estimated_free_energy: float = Field(..., description="estimated size of free energy / uncertanty this cases is the system")
    
class select_policy_fn(BaseModel):
    """Your task: 
    In your response, propose 3-5 multi step policies that will reduce the maximum free energy in the system.
    Finally, pick out the policy you predict will reduce the maximum free energy (estimated_free_energy_reduction * probability_of_success).
    All aspects should be considered from the assistant's point of view.

    Note: The types of polices the assistant can execute successful are speech based in terms of specific questions, statements, chains of conversation. or to pause and wait for the user.
    Note: the types of expected_outcomes should be specific predictions about changes to the perception stream, the hidden_states, the set of beliefs.
    """

    # free_energy_causes: List[FreeEnergy] = Field(..., description="3 to 5 causes of free energy / uncertanty in the system")
    policies: List[Policy] = Field(..., description="3 to 5 multi step policies that will reduce the maximum free energy in the system")
    selected_policy_idx: int = Field(..., description="the index of the policy you predict will reduce the maximum free energy (estimated_free_energy_reduction * probability_of_success)")