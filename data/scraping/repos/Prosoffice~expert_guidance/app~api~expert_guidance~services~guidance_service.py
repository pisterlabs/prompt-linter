from typing import Optional, List, Type, Any
import httpx

from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from app.api.expert_guidance.dto.guidance_dto import GuidanceCreateDto
from app.api.expert_guidance.models.e_guidance_model import ExpertGuidance
from app.api.expert_guidance.services import BaseService
from app.api.expert_guidance.services.ml_service import compute_expert_response


class GuidanceService(BaseService[ExpertGuidance, GuidanceCreateDto]):

    def __init__(self, expert_guidance_model):
        self.kms_url = "http://127.0.0.1:8002"
        self.expert_guidance_model = expert_guidance_model

    def get_all(self, db: Session) -> List[Type[ExpertGuidance]]:
        return db.query(ExpertGuidance).all()

    def get(self, db: Session, id: Any) -> Optional[ExpertGuidance]:
        return db.query(ExpertGuidance).filter(ExpertGuidance.id == id).first()

    def create(self, db: Session, *, obj_in: GuidanceCreateDto) -> ExpertGuidance:
        db_obj = ExpertGuidance(
            user_id=obj_in.user_id,
            query=obj_in.query,
            result=obj_in.result,
            content=obj_in.content,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    async def generate_expert_guidance(self, query, token):
        query_string = query

        # Get All Regulations in Article
        relevant_regulations = await self.get_all_regulations(token)

        legal_dataset = [item["content"] for item in relevant_regulations]

        # Applying Machine Learning to search most relevant according to query
        expert_guidance = compute_expert_response(query_string, legal_dataset)

        # Init article number
        article_number = 0

        for i in relevant_regulations:
            if i['content'] == expert_guidance["response"]:
                article_number = i["article_id"]

        return expert_guidance, article_number

    async def get_articles_via_context(self, context, token):

        try:
            headers = {
                'Authorization': f'Bearer {token}'  # Adding the bearer token to the headers
            }
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.kms_url}/gdpr_articles/{context}", headers=headers)
                data = response.json()
                return data
        except Exception as e:
            print(e)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Could not validate credentials",
            )

    async def get_regulations_by_article(self, id, token):
        try:
            headers = {
                'Authorization': f'Bearer {token}'  # Adding the bearer token to the headers
            }
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.kms_url}/regulations/{id}", headers=headers)
                data = response.json()
                return data
        except Exception as e:
            print(e)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Could not validate credentials",
            )

    async def get_all_regulations(self, token):
        try:
            headers = {
                'Authorization': f'Bearer {token}'  # Adding the bearer token to the headers
            }
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.kms_url}/regulations", headers=headers)
                data = response.json()
                return data
        except Exception as e:
            print(e)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Could not validate credentials",
            )


expertGuidanceService = GuidanceService(ExpertGuidance)

