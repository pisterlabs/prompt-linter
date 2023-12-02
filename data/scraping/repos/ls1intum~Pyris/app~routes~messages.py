from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from guidance._program_executor import SyntaxException
from parsimonious.exceptions import IncompleteParseError

from app.core.custom_exceptions import (
    MissingParameterException,
    InvalidTemplateException,
    InternalServerException,
    InvalidModelException,
)
from app.dependencies import TokenPermissionsValidator
from app.models.dtos import (
    SendMessageRequest,
    SendMessageResponse,
    Content,
    ContentType,
    SendMessageRequestV2,
    SendMessageResponseV2,
)
from app.services.circuit_breaker import CircuitBreaker
from app.services.guidance_wrapper import GuidanceWrapper
from app.config import settings

router = APIRouter(tags=["messages"])


def execute_call(template, preferred_model, parameters) -> dict:
    try:
        model = settings.pyris.llms[preferred_model]
    except ValueError as e:
        raise InvalidModelException(str(e))

    guidance = GuidanceWrapper(
        model=model,
        handlebars=template,
        parameters=parameters,
    )

    try:
        return CircuitBreaker.protected_call(
            func=guidance.query,
            cache_key=preferred_model,
            accepted_exceptions=(
                KeyError,
                SyntaxError,
                SyntaxException,
                IncompleteParseError,
            ),
        )
    except KeyError as e:
        raise MissingParameterException(str(e))
    except (SyntaxError, SyntaxException, IncompleteParseError) as e:
        raise InvalidTemplateException(str(e))
    except Exception as e:
        raise InternalServerException(str(e))


@router.post(
    "/api/v1/messages", dependencies=[Depends(TokenPermissionsValidator())]
)
def send_message(body: SendMessageRequest) -> SendMessageResponse:
    generated_vars = execute_call(
        body.template.content, body.preferred_model, body.parameters
    )

    # V1: Throw an exception if no 'response' variable was generated
    if "response" not in generated_vars:
        raise InternalServerException(
            str(ValueError("The handlebars do not generate 'response'"))
        )

    return SendMessageResponse(
        usedModel=body.preferred_model,
        message=SendMessageResponse.Message(
            sentAt=datetime.now(timezone.utc),
            content=[
                Content(
                    type=ContentType.TEXT,
                    textContent=generated_vars[
                        "response"
                    ],  # V1: only return the 'response' variable
                )
            ],
        ),
    )


@router.post(
    "/api/v2/messages", dependencies=[Depends(TokenPermissionsValidator())]
)
def send_message_v2(body: SendMessageRequestV2) -> SendMessageResponseV2:
    generated_vars = execute_call(
        body.template, body.preferred_model, body.parameters
    )

    return SendMessageResponseV2(
        usedModel=body.preferred_model,
        sentAt=datetime.now(timezone.utc),
        content=generated_vars,  # V2: return all generated variables
    )
