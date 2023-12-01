from fastapi import APIRouter
from app.api.expert_guidance.controllers import guidance_controller


api_router = APIRouter()
api_router.include_router(guidance_controller.router, tags=["Expert Guidance"])
