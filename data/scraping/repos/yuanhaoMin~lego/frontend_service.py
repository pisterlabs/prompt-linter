from chromadb import Client
from config.vectordb_config import FRONTEND_OPERATION_COLLECTION
from constant.frontend_operation_enum import EMPTY_OPERATION, FrontendOperation
from constant.frontend_operation_param_enum import FrontendOperationParam
from fastapi import HTTPException
from langchain.schema import HumanMessage, SystemMessage
from util.json_utils import fix_and_parse_json
from util.openai_utils import completion, embedding


def determine_function(
    operation_text: str, last_operation: str, require_param: bool, client: Client
) -> str:
    confirm_words = ["确认", "提交"]
    cancel_words = ["取消", "返回"]
    if any(word in operation_text for word in confirm_words):
        return {
            "function_name": "submit",
            "function_level": 2,
        }
    elif any(word in operation_text for word in cancel_words):
        return {
            "function_name": "cancel",
            "function_level": 2,
        }

    collection = client.get_collection(FRONTEND_OPERATION_COLLECTION)

    query_embeddings = embedding([operation_text])
    results = collection.query(
        n_results=1,
        query_embeddings=query_embeddings,
        where={
            "$or": [
                {"parent_operation": {"$eq": EMPTY_OPERATION}},
                {"parent_operation": {"$eq": last_operation}},
            ]
        },
    )
    operation = FrontendOperationParam.find_by_function_name(results["ids"][0][0])
    if operation:
        return determine_function_and_param(operation, operation_text)
    else:
        if require_param:
            operation = FrontendOperationParam.find_by_parent_operation(last_operation)
            return determine_function_and_param(operation, operation_text)
        else:
            operation = FrontendOperation.find_by_function_name(results["ids"][0][0])
            return {"function_name": operation.function_name, "function_level": 1}


def determine_function_and_param(
    operation: FrontendOperationParam, operation_text: str
):
    system_message = SystemMessage(content=operation.param_prompt)
    user_message = HumanMessage(content=operation_text)
    parameter_json = completion([system_message, user_message])
    try:
        data = fix_and_parse_json(parameter_json)
        return {
            "function_name": operation.function_name,
            "data": data,
            "function_level": 2,
        }
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse parameters '{parameter_json}' for the operation '{operation.function_name}'",
        )
