import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from app.models.resume import Person
from app.utils.network import Network
from app.prompts.resume_parsing import extractor_function, core_program
from app.utils.load_env import load_env
from typing import List, Any, Dict, Optional
import re
import yaml
import guidance
import logging
import os

logger = logging.getLogger("resume")

GPT_MODEL = "gpt-3.5-turbo-0613"
guidance.llm = guidance.llms.OpenAI(model=GPT_MODEL)


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
async def chat_completion_request(
    messages: List[Dict[str, Any]],
    functions: List[Dict[str, Any]],
    function_call: Dict[str, str],
    model: str = GPT_MODEL,
):
    logger.debug(
        "chat_completion_request with function call to extract metadata")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "functions": functions,
        "function_call": function_call,
    }
    try:
        async with Network().session.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=json_data
        ) as response:
            logger.debug(f"Response received")
            if response.status != 200:
                raise Exception(f"Error extracting metadata")
            result = await response.json()
            return result
    except Exception as e:
        logger.debug(f"Error extracting metadata: {e}")
        return e


functions = [extractor_function]


async def getResumeDetails(text: str) -> Optional[Person]:
    """
    Tries getting resume details using function call api of OpenAI
    """
    messages = []
    messages.append(
        {
            "role": "system",
            "content": "You are a PDF parser who can extract details of person from text of resume delimited by triple backticks. You only extract information present in the text and do not generate any new information not present in the given text.",
        }
    )
    messages.append(
        {
            "role": "user",
            "content": f"resume text: ```{text}```",
        }
    )
    result = await chat_completion_request(
        messages,
        functions=functions,
        function_call={"name": "extract_resume_details"},
        model=GPT_MODEL,
    )

    # if above method returns an Exception
    if isinstance(result, Exception):
        logger.error(f"Error extracting resume using GPT: {result}")
        return None

    logger.debug("now extracting details from response")
    result = result["choices"][0]["message"]["function_call"]["arguments"]
    # just to test locally
    # import json
    # with open("test.json", "w") as f:
    #     json.dump(json.loads(result), f)
    person: Person = Person.from_json_str(result)  # type: ignore
    logger.debug("resume details extracted")

    return person


def fix_missing_commas(json_string: str) -> str:
    """Fix missing commas in JSON string for validity."""
    json_string = re.sub(r'"\n', r'",\n', json_string)
    json_string = re.sub(r"(\d)\n", r"\1,\n", json_string)
    return json_string


"""
Below is using your guidance prompt but the structure returned would be different
"""


async def getResumeUsingGuidance(text: str) -> Any:
    try:
        logger.debug("getting resume using guidance")
        info: Program = await core_program(resume_text=text)  # type: ignore
        data = info["resume_summary"]
        logger.debug(f"got response from GPT. Now parsing response")
        if isinstance(data, str):
            try:
                parsed = yaml.safe_load(data)
            except Exception:
                # ensure all newlines not ending in ",\n" to end in ",\n"
                data = fix_missing_commas(data)
                parsed = yaml.safe_load(data)

            return parsed
        else:
            # there is some error
            logger.error(f"got error from guidance. Error: {data}")
            return QuestionsInfo(questions=[])
    except Exception as e:
        logger.debug(f"Error extracting resume details using guidance: {e}")
        return QuestionsInfo(questions=[])
