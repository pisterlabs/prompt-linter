from fastapi import HTTPException, status
from langchain import schema
from services.world.internal.models.character import Character


class World(Character):
    timestamp: int = 0

    def add_memory(self, text: str) -> None:
        self.memory.append(
            schema.Document(
                page_content=text,
                metadata={
                    "created_at": self.timestamp,
                    "last_accessed_at": self.timestamp,
                    "buffer_idx": len(self.memory),
                },
            )
        )


async def get_world(world_name: str) -> World:
    world = await World.find({"name": world_name}).first_or_none()
    if world is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"world {world_name} not found",
        )
    return world


async def check_world_exist(world_name: str) -> bool:
    """check_world_exist doesn't retrieve the whole world object,
    so it's faster than get_world.
    """
    return await World.find_one({"name": world_name}).exists()
