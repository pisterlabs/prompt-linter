from typing import Any
from fastapi import Depends, APIRouter
from sqlalchemy.orm import Session
from app.api.expert_guidance.dto.guidance_dto import GuidanceCreateDto, Guidance
from app.api.expert_guidance.factories.db_factory import get_db
from app.api.expert_guidance.services.guidance_service import expertGuidanceService
from app.core.security import auth_service
from app.core.utils import AuthUserDto

router = APIRouter()


@router.post("/generate-guide", response_model=[])
async def create_guidance(
        *,
        obj_in: GuidanceCreateDto,
        current_user: AuthUserDto = Depends(auth_service)
) -> Any:
    result, arti_number = await expertGuidanceService.generate_expert_guidance(obj_in.query, current_user["token"])
    result["arti_number"] = arti_number
    return result
