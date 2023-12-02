from fastapi import APIRouter, Depends

from app.dependencies import TokenValidator
from app.models.dtos import LLMStatus, ModelStatus
from app.services.guidance_wrapper import GuidanceWrapper
from app.services.circuit_breaker import CircuitBreaker
from app.config import settings

router = APIRouter()


@router.get("/api/v1/health", dependencies=[Depends(TokenValidator())])
def checkhealth() -> list[ModelStatus]:
    result = []

    for key, model in settings.pyris.llms.items():
        circuit_status = CircuitBreaker.get_status(
            checkhealth_func=GuidanceWrapper(model=model).is_up,
            cache_key=key,
        )
        status = (
            LLMStatus.UP
            if circuit_status == CircuitBreaker.Status.CLOSED
            else LLMStatus.DOWN
        )
        result.append(ModelStatus(model=key, status=status))

    return result
