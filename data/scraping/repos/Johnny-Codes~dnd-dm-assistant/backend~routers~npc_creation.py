from fastapi import (
    HTTPException,
    APIRouter,
    Depends,
)
from models.npc_creation import (
    CreateNPCIn,
    CreateNPCOut,
    UpdateNPCData,
)
from openai_api.npc_creation import npc_creation
from queries.npc_creation import NPCCreationRepo

router = APIRouter()


@router.post("/api/npcs", response_model=CreateNPCOut)
async def create_npc_character(
    model: CreateNPCIn,
    repo: NPCCreationRepo = Depends(),
):
    work = model.work
    add_info = model.additional_information
    npc = npc_creation(work, add_info)
    created_npc = repo.create_npc(npc)
    if created_npc:
        return created_npc
    else:
        raise HTTPException(status_code=500, detail="Failed to create NPC")


@router.get("/api/npcs")
async def get_all_npcs(
    repo: NPCCreationRepo = Depends(),
):
    all_npcs = repo.get_all_npcs()
    return all_npcs


@router.get("/api/npcs/{npc_id}", response_model=CreateNPCOut)
async def get_npc(
    npc_id: int,
    repo: NPCCreationRepo = Depends(),
):
    npc = repo.get_npc(npc_id)
    return npc


@router.delete("/api/npcs/{npc_id}")
async def delete_npc(
    npc_id: int,
    repo: NPCCreationRepo = Depends(),
):
    npc = repo.delete_npc(npc_id)
    return npc


@router.put("/api/npcs/{npc_id}", response_model=CreateNPCOut)
async def update_npc(
    npc_id: int,
    model: UpdateNPCData,
    repo: NPCCreationRepo = Depends(),
):
    model.id = npc_id
    updated_npc = repo.update_npc(model)

    if updated_npc is None:
        raise HTTPException(status_code=404, detail="NPC not found")
    elif updated_npc is False:
        raise HTTPException(status_code=500, detail="Failed to update NPC")
    else:
        return updated_npc
