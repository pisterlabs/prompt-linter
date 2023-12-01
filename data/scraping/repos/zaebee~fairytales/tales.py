import logging
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps
from app.services import cohere
from app.services import stability

router = APIRouter()
logger = logging.getLogger('uvicorn')


@router.post('/', response_model=schemas.TaleBase)
async def create_tale(
    *, tale_in: schemas.TaleCreate,
) -> Any:
    """
    Create new tale.
    """
    tale_prompt = await cohere.TalePrompt.create(tale_in.log_line)
    if tale_in.heroes:
        descriptions = [hero.description for hero in tale_in.heroes]
        names = [hero.name for hero in tale_in.heroes]
        tale_prompt.heroes = {0: {'descriptions': descriptions, 'names': names}}
    if tale_in.structure and tale_in.structure.parts:
        parts = [f'{part.name.upper()}: {part.text}'
            for part in tale_in.structure.parts]
        tale_prompt.structures = {0: parts}
    response = await tale_prompt.get_tale(
        structure=0, heroes=0,
        temperature=tale_in.temperature, max_tokens=tale_in.max_tokens)
    await tale_prompt.close()
    logger.info('Generated tale:\n %s', response)
    tale_in.stories = [schemas.Story(text=text) for text in response]
    return tale_in


@router.post('/heroes', response_model=list[schemas.HeroSet])
async def create_heroes(
    *, tale_in: schemas.TaleCreate,
) -> Any:
    """
    Create new heroes.
    """
    logger.info('Passed tale:%s', tale_in)
    tale_prompt = await cohere.TalePrompt.create(tale_in.log_line)
    response = await tale_prompt.get_heroes(
        temperature=tale_in.temperature,
        max_tokens=tale_in.max_tokens)
    logger.info('Generated heroes:\n %s', response)
    await tale_prompt.close()
    return [schemas.HeroSet(heroes=heroes) for heroes in response]


@router.post('/structures', response_model=list[schemas.Structure])
async def create_structures(
    *, tale_in: schemas.TaleCreate,
) -> Any:
    """
    Create new structures.
    """
    logger.info('Passed tale:\n %s', tale_in)
    tale_prompt = await cohere.TalePrompt.create(tale_in.log_line)
    descriptions = [hero.description for hero in tale_in.heroes]
    tale_prompt.heroes = {0: {'descriptions': descriptions}}
    response = await tale_prompt.get_structure(
        heroes=0, temperature=tale_in.temperature,
        max_tokens=tale_in.max_tokens)
    logger.info('Generated structures:\n %s', response)
    await tale_prompt.close()
    return [schemas.Structure(parts=item) for item in response.values()]



@router.post('/portraits', response_model=list[schemas.Portrait])
def create_portraits(
    *, image_in: schemas.PortraitCreate,
) -> Any:
    """
    Create hero portraits.
    """
    image_prompt = stability.StabilityPrompt()
    response = image_prompt.generate_character(
        image_in.hero_id, image_in.prompt, style=image_in.style)
    logger.info('Generated images:\n%s', response)
    return [schemas.Portrait(**item) for item in response]



@router.post('/images', response_model=list[schemas.Scene])
def create_images(
    *, image_in: schemas.SceneCreate,
) -> Any:
    """
    Create scene images.
    """
    image_prompt = stability.StabilityPrompt()
    response = image_prompt.generate_scene(
        image_in.scene_id, image_in.prompt, style=image_in.style)
    logger.info('Generated images:\n%s', response)
    return [schemas.Scene(**item) for item in response]
