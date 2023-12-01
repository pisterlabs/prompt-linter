"""
Module to suggest evals for the user to run based on prompt deltas.
"""
import difflib
import hashlib
import json
from typing import Callable, Optional

from copy import deepcopy
from dotenv import load_dotenv

import logging
import re
from langchain.adapters.openai import convert_openai_messages
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.callbacks.manager import trace_as_chain_group

llm = ChatOpenAI(model="gpt-4")

load_dotenv()

# import os

# BASE_URL = os.getenv("AZURE_API_BASE")
# API_KEY = os.getenv("AZURE_API_KEY")
# DEPLOYMENT_NAME = "gpt-4"
# llm = AzureChatOpenAI(
#     openai_api_base=BASE_URL,
#     openai_api_version=os.getenv("AZURE_API_VERSION"),
#     deployment_name=DEPLOYMENT_NAME,
#     openai_api_key=API_KEY,
#     openai_api_type="azure",
# )


def show_diff(template_1: str, template_2: str):
    diff = difflib.ndiff(template_1.splitlines(), template_2.splitlines())
    return "\n".join(diff)


def show_readable_diff(template_1: str, template_2: str):
    diff = list(
        difflib.ndiff(template_1.splitlines(), template_2.splitlines())
    )

    # Processing the diff list for readable format
    added_lines = []
    deleted_lines = []

    for line in diff:
        if line.startswith("- "):
            deleted_lines.append(line)
        if line.startswith("+ "):
            added_lines.append(line)

    output = []

    if deleted_lines:
        output.append("Deleted lines:")
        output.extend(deleted_lines)

    if added_lines:
        output.append("Added lines:")
        output.extend(added_lines)

    return "\n".join(output)


CATEGORIZE_TEMPLATE = """I need your assistance in identifying and categorizing any new instructions I've added to my prompt template, that add requirements for LLM pipeline responses to satisfy.

**First Prompt Template**: 
```
{template_1}
```

**Second Prompt Template** (Updated):
```
{template_2}
```

**Changes Highlighted**:
```
{diff}
```

Please focus your analysis on the newly added instructions in the updated prompt template. Use the categories listed below to describe the changes::

- **Structural**:
  - **Format Instruction**: Have any new specifications been added regarding the expected response format, such as a list, JSON, Markdown, or HTML?
  - **Example Demonstration**: Are there any new examples provided to demonstrate the desired response format, including specific headers, keys, or structures?
  - **Prompt Rephrasing (not a new instruction)**: Has the prompt been rephrased slightly to clarify the task, maintaining the same overall semantic meaning?

- **Content**:
  - **Workflow Description**: Have more detailed steps on how to perform the task been newly added?
  - **Data Placeholders**: Have any new data sources or context been inserted in placeholders for the LLM to consider?
  - **Quantity Instruction**: Have there been new specifications added regarding the number of items of a certain type in the response, such as “at least”, “at most”, or an exact number?
  - **Inclusion**: Are there new keywords that every future LLM response should now include?
  - **Exclusion**: Have any new keywords been specified that should be excluded from all future LLM responses?
  - **Qualitative Assessment**: Are there new qualitative criteria for assessing good responses, including specific requirements for length, tone, or style?

**Expected Output Structure**:

```json
{{
  "Structural": {{
    "FormatInstruction": "Describe new format specifications (if any)",
    "ExampleDemonstration": "Describe new example structure (if any)",
    "PromptRephrasing": "Change description (if any)"
  }},
  "Content": {{
    "WorkflowDescription": "Describe added workflow steps (if any)",
    "DataPlaceholders": "Describe added data sources or context (if any)",
    "QuantityInstruction": "Describe new item quantity specifications (if any)",
    "Inclusion": "State new keywords for LLM to include in all responses (if any)",
    "Exclusion": "State new keywords for LLM to exclude from all responses (if any)",
    "QualitativeAssessment": "Describe new qualitative criteria of a good LLM response (if any)"
  }}
}}
```

Please fill out this structure based on your analysis of the newly added instructions. For any categories without changes, please write "No change." Remember, at this stage, we are focusing on identifying additions to the prompt template, not deletions."""

SUGGEST_EVAL_TEMPLATE = """Please use this JSON structure, detailing the newly added instructions to the LLM prompt template, to design at least one evaluation function for each applicable change.

**Requirements and Guidelines:**

1. Limit the use of libraries in your functions to json, numpy, pandas, re, and other standard Python libraries.
2. For complex evaluations where pattern matching or logical checks are insufficient, you may use the `ask_expert` function. This function sends a specific yes-or-no question to a human expert and returns a boolean value.
3. All evaluation functions should return a binary True or False value.
4. All evaluation functions should have a descriptive name and comment explaining the purpose of the function.
5. When creating functions for QualitativeAssesment prompt additions, target the specific criteria added to the prompt rather than creating generic evaluations. For instance, if the criteria specifies a "concise response", the function might check the length of the response and decide whether it's concise. Create a different function for each qualitative criteria, even if there are multiple criteria in the same prompt edit.
6. Use the following template for each function, only accepting a formatted LLM prompt (filled with values in the placeholders) and response as arguments:

**Function Signature**:
```python
def evaluation_function_name(prompt: str, response: str) -> bool:
    # Your implementation here
```

**Evaluation Categories and Example Functions:**

Below are examples of functions for each type of change you might encounter, based on the JSON structure provided:

{example_evals}

**Important Notes:**

- If writing a conditional based on keywords in the formatted prompt, make sure the keywords aren't always present in the prompt template. For instance, if the prompt template always contains the word "wedding", don't write a function that checks if the response contains the word "wedding"--use a phrase like "my wedding" to check in the conditional. 
- Customize the provided function templates based on the actual criteria specified in the given JSON output of changes. You'll need to adjust the specifics (like the exact phrases or counts) based on the actual criteria I've added to my prompts. Make sure each function has a descriptive name and comment explaining the purpose of the function.
- Do not create evaluation functions for changes categorized under "PromptRephrasing" or "DataPlaceholders".
- Ensure that each function serves as a standalone validation tool to run on every response to the prompt. Each function should be correct and complete, and should not rely on other non-library functions to run."""

EXAMPLE_EVALS = {
    "FormatInstruction": """**Format Instruction**:
    - If the desired format is JSON:
    ```python
    def evaluate_json_format(prompt: str, response: str) -> bool:
        try:
            json.loads(response)
            return True
        except json.JSONDecodeError:
            return False
    ```
    - If the desired format is a list:
    ```python
    def evaluate_list_format(prompt: str, response: str) -> bool:
        # Check if response starts with a bullet point or number
        return response.startswith("- ") or response.startswith("1. ")
    ```
    """,
    "ExampleDemonstration": """**Example Demonstration**:
    ```python
    def check_example_demonstration(prompt: str, response: str) -> bool:
        # Suppose the example demonstration is markdown and has specific headers
        # of "First Header" and "Second Header"
        return "# First Header" in response and "# Second Header" in response
    ```
    """,
    "QuantityInstruction": """**Quantity Instruction**:
    ```python
    def evaluate_num_distinct_words(prompt: str, response: str) -> bool:
        # Suppose responses should contain at least 3 distinct words
        distinct_word_count = len(set(response.split()))
        return distinct_word_count >= 3
    ```
    """,
    "Inclusion": """**Inclusion**:
    ```python
    def check_includes_color(prompt: str, response: str) -> bool:
        # Suppose the response should include some color in the rainbow
        colors = ["red", "orange", "yellow", "green", "blue", "purple", "indigo"]
    
        return any(color in response for color in colors)
    ```
    """,
    "Exclusion": """**Exclusion**:
    ```python
    def check_excludes_white(prompt: str, response: str) -> bool:
        # Suppose the prompt template instructs not to include the color
        # white for queries related to wedding dresses for guests
        if "my wedding" in prompt:
            return "white" not in response.lower()
        else:
            return True
    ```
    """,
    "QualitativeAssessment": """**Qualitative Assessment**:
    - If the desired length is concise:
    ```python
    def evaluate_concise(prompt: str, response: str) -> bool:
        # Suppose the response should be less than 50 characters
        return len(response) < 50
    ```
    - If the desired tone is positive:
    ```python
    def evaluate_tone(prompt: str, response: str) -> bool:
        return ask_expert(f"Is the tone of the response \{response\} positive?")
    ```
    """,
}

RENDER_DIFF_TEMPLATE = """Please use this JSON structure, detailing the newly added instructions to the LLM prompt template, to render the second prompt template with the changes highlighted. You should return the same second prompt template, but wrap each identified change based on the JSON structure of changes in its tag. Make sure each change has opening and closing tags (e.g., <FormatInstruction></FormatInstruction>). Category tags should not be nested. Your answer should start with <FormattedPromptTemplate> and end with </FormattedPromptTemplate>"""

# Hash all prompts into a single string
combined_content = (
    CATEGORIZE_TEMPLATE
    + SUGGEST_EVAL_TEMPLATE
    + RENDER_DIFF_TEMPLATE
    + str(EXAMPLE_EVALS)
).encode()
hash_object = hashlib.sha256(combined_content)
PIPELINE_PROMPT_HASH = hash_object.hexdigest()


def get_suggest_eval_prompt(changes_flagged):
    # See which keys have been flagged
    examples_to_include = [
        f"{str(i)}. {EXAMPLE_EVALS[key]}"
        for i, key in enumerate(changes_flagged)
        if key in EXAMPLE_EVALS
    ]
    # Format the prompt
    examples_to_include_str = "\n".join(examples_to_include)
    return SUGGEST_EVAL_TEMPLATE.format(example_evals=examples_to_include_str)


async def suggest_evals(
    template_1: str,
    template_2: str,
    source: str,
    characterize_callback: Optional[Callable[[str], None]] = None,
    eval_prediction_callback: Optional[Callable[[str], None]] = None,
):
    """Suggest evals for the user to run based on prompt deltas."""
    with trace_as_chain_group(
        "suggest_evals",
        inputs={
            "template_1": template_1,
            "template_2": template_2,
        },
        tags=[source, PIPELINE_PROMPT_HASH],
    ) as cb:
        # If the templates are the same, return []
        if template_1 == template_2:
            # Send callback that there is no change
            characterize_callback("Prompt templates are the same.")

            cb.on_chain_end(
                {
                    "eval_functions": [],
                    "messages": [],
                    "rendered_diff": None,
                }
            )
            return [], [], None

        template_1_pretty = template_1 if template_1 != "" else "Empty string"
        template_2_pretty = template_2 if template_2 != "" else "Empty string"

        diff = show_readable_diff(template_1, template_2)
        messages = [
            {
                "content": "You are an expert in Python and prompting large language models (LLMs). You are assisting me, a prompt engineer, build and monitor an LLM pipeline. An LLM pipeline accepts a prompt, with some instructions, and uses an LLM to generate a response to the prompt. A prompt engineer writes a prompt template, with placeholders, that will get formatted with different variables at pipeline runtime. Typically as prompt engineers test a pipeline, we observe that some responses are not good, so we add new instructions to the prompt template to prevent against these failure modes.",
                "role": "system",
            },
            {
                "content": CATEGORIZE_TEMPLATE.format(
                    template_1=template_1_pretty,
                    template_2=template_2_pretty,
                    diff=diff,
                ),
                "role": "user",
            },
        ]

        # First characterize the deltas
        try:
            lc_messages = convert_openai_messages(messages)
            char_response = llm.astream(lc_messages, {"callbacks": cb})

            logging.debug("Determining prompt deltas...")
            collected_messages = []
            # iterate through the stream of events
            async for chunk in char_response:
                if characterize_callback:
                    characterize_callback(chunk.content)
                collected_messages.append(chunk.content)

            logging.info("")
            reply = "".join(collected_messages)

        except Exception as e:
            logging.error(f"Error getting deltas: {e}")
            cb.on_chain_end(
                {
                    "eval_functions": [],
                    "messages": messages,
                    "rendered_diff": None,
                }
            )
            return [], messages, None

        # Parse the reply's json from ```json ... ```
        reply_json = None
        try:
            pattern = r"```json(.*?)```"
            matches = re.findall(pattern, reply, re.DOTALL)
            reply_json = json.loads(matches[0])
        except Exception as e:
            logging.error(f"Error parsing json: {e}")
            messages.append({"content": reply, "role": "assistant"})
            cb.on_chain_end(
                {
                    "eval_functions": [],
                    "messages": messages,
                    "rendered_diff": None,
                }
            )
            return [], messages, None

        # Look for any changes
        changes_made = []
        for key in reply_json.get("Structural", {}):
            if not isinstance(reply_json["Structural"][key], str):
                continue

            if reply_json["Structural"][key].lower() != "no change":
                changes_made.append(key)
        for key in reply_json.get("Content", {}):
            if not isinstance(reply_json["Content"][key], str):
                continue

            if reply_json["Content"][key].lower() != "no change":
                changes_made.append(key)

        # Remove promptrephrasing and dataorcontextaddition
        if "PromptRephrasing" in changes_made:
            changes_made.remove("PromptRephrasing")
        # if "WorkflowDescription" in changes_made:
        #     changes_made.remove("WorkflowDescription")
        if "DataPlaceholders" in changes_made:
            changes_made.remove("DataPlaceholders")

        # Render the diff
        try:
            diff_render_messages = deepcopy(messages)
            diff_render_messages.append(
                {"content": RENDER_DIFF_TEMPLATE, "role": "user"}
            )
            diff_render_response = await llm.ainvoke(
                convert_openai_messages(diff_render_messages)
            )
            diff_render_response = diff_render_response.content

            # Extract whatever is in ```<FormattedPromptTemplate> tags
            pattern = (
                r"<FormattedPromptTemplate>(.*?)</FormattedPromptTemplate>"
            )
            matches = re.findall(pattern, diff_render_response, re.DOTALL)
            diff_render_response = matches[0]

        except Exception as e:
            logging.error(f"Error rendering diff: {e}")
            diff_render_response = None

        # If there are no changes, return []
        if not changes_made:
            cb.on_chain_end(
                {
                    "eval_functions": [],
                    "messages": messages,
                    "rendered_diff": diff_render_response,
                }
            )
            return [], messages, diff_render_response

        messages.append(
            {
                "content": reply,
                "role": "assistant",
            }
        )

        # See if there are any deltas that bring upon evals
        eval_functions = []
        # Then extract the evals
        messages.append(
            {"content": get_suggest_eval_prompt(changes_made), "role": "user"}
        )
        logging.debug("Generating evals...")
        lc_messages = convert_openai_messages(messages)
        eval_response_stream = llm.astream(lc_messages, {"callbacks": cb})
        eval_response = []
        async for chunk in eval_response_stream:
            if eval_prediction_callback:
                eval_prediction_callback(chunk.content)
            eval_response.append(chunk.content)
        eval_prediction_callback(None)
        eval_response_content = "".join(eval_response)
        messages.append(
            {"content": eval_response_content, "role": "assistant"}
        )

        # Look for the evals in the response as any instance of ```python ```
        pattern = r"^\s*```python\s+(.*?def.*?)(?=\n\s*```)"  # match any def with leading whitespace
        matches = re.findall(
            pattern, eval_response_content, re.DOTALL | re.MULTILINE
        )

        # Get longest match
        for match in matches:
            match = match.strip()

            try:
                # Replace `ask_expert` with a call to an llm function
                needs_llm = False
                function_str = match
                if "ask_expert" in function_str:
                    function_str = function_str.replace(
                        "ask_expert", "ask_llm"
                    )
                    needs_llm = True

                # Add the function to the list of eval functions
                eval_functions.append(
                    {"code": function_str, "needs_llm": needs_llm}
                )

            except:
                logging.error(f"Error parsing code: {match}")
        cb.on_chain_end(
            {
                "eval_functions": eval_functions,
                "messages": messages,
                "rendered_diff": diff_render_response,
            },
        )
        return eval_functions, messages, diff_render_response


if __name__ == "__main__":
    import asyncio

    ef, m = asyncio.run(
        suggest_evals(
            "You ar ea helpful agent",
            "You are a helpful agent. Respond in less than 3 words",
        )
    )
    print(ef, m)
