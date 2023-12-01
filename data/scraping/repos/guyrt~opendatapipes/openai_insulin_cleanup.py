"""Cleanup insulin records."""
import json
import openai
from typing import List, Dict, Any, AnyStr

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient



drug_prompt = """
For each string below, you must output either "Lantus" or "Novalog". Never output an string other than Lantus or Novalog. Output exactly one answer for every input.

Here are some examples:
Inputs:
1: l
2: L
3: lantis
4: n
5: N
6: novalo

Outputs:
1: Lantus
2: Lantus
3: Lantus
4: Novalog
5: Novalog
6: Novalog

Inputs:
{inputs}

Outputs:

"""


notes_cleanup = """
Each line of input is a raw human input. Your job is to extract several pieces of information into a structured json output. The information you should extract includes:
* Was this a correction dose? Example strings are "correct at" and "control". This is Boolean.
* Location. This is always either 'butt' or 'stomach' or 'split'. If you aren't sure or the input doesn't say, then say 'stomach'. If the input says "side" then return "stomach". Only use split if there are two numbers added together.
* "new_pen" if there was a new pen? The input will always say whether there was a new pen. This is Boolean. Only include in the output if the value is "true". If there was a new pen, the input will always say that there was a new pen. Do not output "true" unless the prompt says there is a new pen.
* "split" if there are two numbers added like "3+4". In this case return "split": {{"stomach": 3, "butt": 4}}. Stomach will always be the first number. Butt will always be the second number.

Outputs should always be in json format.

Here are some examples:
Inputs:
1: "Tight control at 175"
2: "Side. New Pen"
3: "Rice. Topped at 110 them long low. Try 2? None?"
4: "2+4. Should have been 3+4"
5: 45 carbs from 89

Outputs:
1: {{"correction": "true",  "location": "stomach"}}
2: {{"correction": "false", "location": "stomach", "new_pen": "true"}}
3: {{"correction": "false", "location": "stomach"}}
4: {{"correction": "false", "location": "split", "split": {{"stomach": 2, "butt": 4}}}}
5: {{"correction": "false", "location": "stomach"}}


Inputs:
{inputs}

Outputs:
"""


def prep_openai_from_key(api_key):
    openai.api_key = api_key
    #print(f"Prepped openai and seeing models {openai.Model.list()}")


class OpenAIWrapper(object):

    def __init__(self) -> None:
        self._running_cost = {
            "completion_tokens": 734,
            "prompt_tokens": 843,
            "total_tokens": 1577
        }

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        print(self._running_cost)

    def add_cost(self, usage):
        for k in self._running_cost.keys():
            self._running_cost[k] += usage[k]

    def run_openai_to_text(self, full_prompt : AnyStr) -> AnyStr:
        """Wrapper to eventually handle openai retry"""
        response = openai.Completion.create(
            model='text-davinci-003',
            prompt=full_prompt,
            temperature=1.0,
            max_tokens=3000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        self.add_cost(response.usage)
        return response.choices[0].text
        

def prep_openai_from_keyvault(keyvault_url):
    """Get the openai api key to use and configure"""
    print(f"Getting secrets from {keyvault_url}")
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=keyvault_url, credential=credential)
    api_key = secret_client.get_secret("openai-api")  # should set this to be your secret key
    print(f"Key: |{api_key.value}|")
    prep_openai_from_key(api_key.value)


def get_drug_conversions(drug_types_raw):
    """My drug names are a mess. Get GPT to guess what each string means then apply."""
    drug_inputs = [f"{i}: {r}" for i, r in zip(range(1, len(drug_types_raw)+1), drug_types_raw)]

    full_prompt = drug_prompt.format(inputs='\n'.join(drug_inputs))
    with OpenAIWrapper() as openai_runner:
        raw_text = openai_runner.run_openai_to_text(full_prompt)
    outputs = [s.split(' ')[1] for s in raw_text.split("\n")]
    
    if len(drug_types_raw) != len(outputs):
        raise Exception(f"Different number of outputs than inputs: {len(drug_types_raw)} inputs and {len(outputs)} outputs")

    ret_val = {k: v for k, v in zip(drug_types_raw, outputs)}

    for k, v in ret_val.items():
        if v not in ("Lantus", "Novalog"):
            raise Exception(f"Value {k} translated to {v} which is not allowed.")
    return ret_val, "1.0.0_tdv003"


def convert_notes(all_rows) -> List[Dict[Any, Any]]:
    """Convert all raw notes to structured outputs."""
    group_size = 40
    row_groups = [all_rows[i:i+group_size] for i in range(0, len(all_rows), group_size)]

    with OpenAIWrapper() as openai_wrapper:

        for group in row_groups:
            input_rows = [f"{i}: {g['Notes']}" for i, g in zip(range(1, len(group)+1), group)]
            full_prompt = notes_cleanup.format(inputs='\n'.join(input_rows))
            response = openai_wrapper.run_openai_to_text(full_prompt)

            print(f"Successful GPT run")

            output_rows = response.split('\n')
            if len(output_rows) != len(group):
                raise Exception(f"Incorrect number of notes rows: expected {len(group)} got {len(output_rows)}")

            parsed_rows = []
            for row in output_rows:
                try:
                    data = row.split(' ', 1)[1].strip()
                    if data:
                        parsed_rows.append(json.loads(row.split(' ', 1)[1]))
                    else:
                        parsed_rows.append({'location': 'stomach', 'correction': 'false'})
                except json.JSONDecodeError:
                    # example failure:  '8: {"correction": "true", "location": "stomach"}', '9: ', '10: {"correction": "false", "location": "stomach"}'
                    # Problem was 'Notes': '45 carbs from 89'
                    import ipdb; ipdb.set_trace()
                    raise

            for row, notes in zip(group, parsed_rows):
                row['ParsedNotesVersion'] = "1.0.0_tdv003"
                row['ParsedNodes'] = notes
                yield row


guidance_prompt = """Each line of input is a raw human input. Your job is to extract several pieces of information into a structured json output. The information you should extract includes:
* Was this a correction dose? Example strings are "correct at" and "control". This is Boolean.
* Location. This is always either 'butt' or 'stomach' or 'split'. If you aren't sure or the input doesn't say, then say 'stomach'. If the input says "side" then return "stomach". Only use split if there are two numbers added together.
* "new_pen" if there was a new pen? The input will always say whether there was a new pen. This is Boolean. Only include in the output if the value is "true". If there was a new pen, the input will always say that there was a new pen. Do not output "true" unless the prompt says there is a new pen.
* "split" if there are two numbers added like "3+4". In this case return "split": {"stomach": 3, "butt": 4}. Stomach will always be the first number. Butt will always be the second number.

Here is an example:
Input:
"Tight control at 175"
Output:
```json
{
    "correction": "true",  
    "location": "stomach",
    "split": {
        "stomach": -1,
        "butt": -1
    },
    "new_pen": "false"
}
```

Input:
"Side. New Pen"
Output:
```json
{
    "correction": "false", 
    "location": "stomach",
    "split": {
        "stomach": -1,
        "butt": -1
    },
    "new_pen": "true"
}
```

Input:
"Rice. Topped at 110 them long low. Try 2? None?"
Output:
```json
{
    "correction": "false", 
    "location": "stomach",
    "split": {
        "stomach": -1,
        "butt": -1
    },
    "new_pen": "false"
}
```json

Input:
"2+4. Should have been 3+4"
Output:
```json
{
    "correction": "false", 
    "location": "split", 
    "split": 
        "stomach": 2, 
        "butt": 4
    },
    "new_pen": "false"
}
```

Input:
{{input}}
Output:
```json
{
    "correction": "{{#select 'correction'}}true{{or}}false{{/select}}",
    "location": "{{#select 'location'}}stomach{{or}}butt{{or}}split{{/select}}",
    "split": {
        "stomach": {{gen 'stomach'}},
        "butt": {{gen 'butt'}}
    },
    "new_pen": "{{#select}}true{{or}}false{{/select}}"
}```

"""

# this fails in OpenAI: "butt": {{gen 'butt' pattern='[0-9]+'}}

def convert_notes_with_ms_guidance(all_rows) -> List[Dict[Any, Any]]:
    """Convert all raw notes to structured outputs.
    This method uses github.com/microsoft/guidance
    """
    import guidance
    guidance.llm = guidance.llms.OpenAI("text-davinci-003")

    prompt = guidance(guidance_prompt)
    import ipdb; ipdb.set_trace()
    output = prompt(input=all_rows[0]['Notes'])    
    print(output)

