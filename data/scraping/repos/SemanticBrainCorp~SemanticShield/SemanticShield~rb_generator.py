import openai

from pydantic import BaseModel, Field
from typing import List
from guardrails.validators import ValidLength

import guardrails as gd


guard_openai_initialized = False
def init_openai_key():
    global guard_openai_initialized
    if not guard_openai_initialized:
        if 'OPENAI_API_KEY' not in os.environ:
            raise APIKEYException(f"OPENAI key not set")
        openai.api_key = os.environ['OPENAI_API_KEY']
        guard_openai_initialized = True

def generate_for_role(role_config_df, role, user_content):
    system_content = "You are a communication specialist, skilled at helping generate marketing, sales, support and other content.\
        You generate content between 400 to 600 characters in length"

    index = role_config_df[role_config_df['role'] == role].index[0]

    columns=['role', 'sys_content_file', 'min_len', 'max_len', 'lang_filter', 'fail_action']
    print("What is the value of this?", index, role_config_df.iloc[index]['role'])

    with open(role_config_df.iloc[index]['sys_content_file']) as f:
        key_instruction = f.read()

    prefix = "You are only capable of communicating in text. "
    suffix = " You generate messages between {} and {} characters in length.".format(role_config_df.iloc[index]['min_len'], role_config_df.iloc[index]['max_len'])
    system_content = prefix + key_instruction + suffix

    class GenerateMessage(BaseModel):
        role: str = Field(description="Is an assistant")
        content: str = Field(
            description="Is content created to help marketing, sales and support staff communicate with customers or prospects",
            validators=[ValidLength(min=role_config_df.iloc[index]['min_len'], max=role_config_df.iloc[index]['max_len'], on_fail='reask')]
        )

    # From pydantic:
    guard = gd.Guard.from_pydantic(output_class=GenerateMessage, instructions=system_content, prompt=user_content)

    # Wrap the OpenAI API call with the `guard` object
    raw_llm_output, validated_output = guard(
        openai.ChatCompletion.create,
        model="gpt-3.5-turbo",
        max_tokens=2048,
        temperature=0.3,
    )

    return validated_output['content']