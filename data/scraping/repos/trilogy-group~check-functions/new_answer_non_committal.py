import json
import pathlib
from typing import Dict, Union, Optional, List, Any

import openai
from openai.error import OpenAIError
from pydantic import BaseModel, ValidationError

from prompt import detect_noncommittal_response
from utils.logger import get_logger
from utils.secret_manager import get_secret


logger = get_logger(__name__)

class Feedback(BaseModel):
    rating: int
    alternative_answer: Optional[List[str]] = None
    comment: Optional[str] = None

class QAPair(BaseModel):
    id: str
    question: str
    answer: str
    feedback: Optional[Feedback] = None
    version: str
    sources: List[str] = []

class LLMToolkitStdCheckInputSchema(BaseModel):
    old_qa_pair: QAPair
    new_qa_pair: QAPair

class LLMToolkitStdCheckOutputSchema(BaseModel):
    '''
    result should ideally have a type of `Union[str, int, float, bool]`
    But there is a weird bug where this converts the dictionary values to strings while loading them
    '''
    result: Dict[str, Any]

class ErrorSchema(BaseModel):
    message: str
    reason: str

class OutputSchema(BaseModel):
    statusCode: int
    body: Union[LLMToolkitStdCheckOutputSchema, ErrorSchema]
def lambda_handler(event: dict, context: dict) -> dict:
    '''
    This function assesses whether or not the old answer is non committal. It conforms to the LLM Toolkit Standard Check API defined here: https://github.com/trilogy-group/llm_toolkit_api/blob/3fe8805e210554b616c47a60216addb01ea14cff/runtime/chalicelib/schema/evaluation.py#L138
    TODO: Replace the above link to a readme once merged
    '''
    logger.info(f"Going to check if new answer is non committal for {event=}")

    try:
        input_data = LLMToolkitStdCheckInputSchema(**event)
    except ValidationError  as ve:
        response = ErrorSchema(
            message="Bad Request Body",
            reason=str(ve),
        )
        return OutputSchema(statusCode=400, body=response).dict()

    try:
        secrets = get_secret()
    except Exception as e:
        logger.error(f"Failed to retrieve secrets: {e}")
        response = ErrorSchema(
            message="Internal Server Error",
            reason=str(e),
        )
        return OutputSchema(statusCode=500, body=response).dict()
    
    return do(
        openai_api_key=secrets["OPENAI_API_KEY"], 
        input_data=input_data,
        prompt_path="prompt.json",
        key="new_answer_non_committal"
    ).dict()

def do(openai_api_key: str, input_data: LLMToolkitStdCheckInputSchema, prompt_path: str, key: str)->OutputSchema:
    user_prompt, system_prompt = detect_noncommittal_response(
        question=input_data.new_qa_pair.question,
        answer = input_data.new_qa_pair.answer,
        prompt_path = str(pathlib.Path(__file__).parent.resolve().joinpath(prompt_path)),
        key=key
    )
    try:
        openai_model: str = 'gpt-3.5-turbo'
        check_result = make_llm_call(openai_api_key, openai_model, user_prompt=user_prompt, system_prompt=system_prompt)
    except OpenAIError as e:
        logger.error(f"Failed to make call to openai: {e}")
        response = ErrorSchema(
            message="Internal Server Error",
            reason=str(e),
        )
        return OutputSchema(statusCode=500, body=response)

    try:
        check_dictionary = json.loads(check_result)
    except json.decoder.JSONDecodeError as e:
        logger.error(f"Failed while trying to parse response {check_result}: {e}")
        return OutputSchema(statusCode=500, body=ErrorSchema(
            message="Check result not valid json",
            reason=str(e),
        ))
    
    try:
        return OutputSchema(statusCode=200, body=LLMToolkitStdCheckOutputSchema(
            result = check_dictionary
        ))
    except ValidationError as e:
        logger.error(f"Failed while trying to set output to {check_dictionary=}")
        return OutputSchema(statusCode=500, body=ErrorSchema(
            message="Check result not valid schema",
            reason=str(e),
        ))


        
def make_llm_call(openai_api_key: str, openai_model: str, user_prompt: str, system_prompt: str, temperature: int = 0)->str:
    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model=openai_model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=0,
    )
    return response['choices'][0]['message']['content'] # type: ignore