from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Response
from starlette import status

from app.api.dependencies.items import (
    check_item_modification_permissions,
    get_item_by_slug_from_path,
    get_items_filters,
)
from app.api.dependencies.authentication import get_current_user_authorizer
from app.api.dependencies.database import get_repository
from app.db.repositories.items import ItemsRepository
from app.models.domain.items import Item
from app.models.domain.users import User
from app.models.schemas.items import (
    ItemForResponse,
    ItemInCreate,
    ItemInResponse,
    ItemInUpdate,
    ItemsFilters,
    ListOfItemsInResponse,
)
from app.resources import strings
from app.services.items import check_item_exists, get_slug_for_item
from app.services.event import send_event

router = APIRouter()

import os
import openai
openai.api_key = os.getenv('OPENAI_API_KEY')
#openai.api_key = 

@router.get("", response_model=ListOfItemsInResponse, name="items:list-items")
async def list_items(
    items_filters: ItemsFilters = Depends(get_items_filters),
    user: Optional[User] = Depends(get_current_user_authorizer(required=False)),
    items_repo: ItemsRepository = Depends(get_repository(ItemsRepository)),
) -> ListOfItemsInResponse:
    items = await items_repo.filter_items(
        tag=items_filters.tag,
        seller=items_filters.seller,
        favorited=items_filters.favorited,
        limit=items_filters.limit,
        offset=items_filters.offset,
        requested_user=user,
    )
    items_for_response = [
        ItemForResponse.from_orm(item) for item in items
    ]
    return ListOfItemsInResponse(
        items=items_for_response,
        items_count=len(items),
    )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=ItemInResponse,
    name="items:create-item",
)
async def create_new_item(
    item_create: ItemInCreate = Body(..., embed=True, alias="item"),
    user: User = Depends(get_current_user_authorizer()),
    items_repo: ItemsRepository = Depends(get_repository(ItemsRepository)),
) -> ItemInResponse:
    slug = get_slug_for_item(item_create.title)
    if await check_item_exists(items_repo, slug):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=strings.ITEM_ALREADY_EXISTS,
        )
    if not item_create.image:
        response = openai.Image.create(
            prompt=item_create.title,
            n=1,
            size='256x256'
        )
        item_create.image = response['data'][0]['url']
    item = await items_repo.create_item(
        slug=slug,
        title=item_create.title,
        description=item_create.description,
        body=item_create.body,
        seller=user,
        tags=item_create.tags,
        image=item_create.image
    )
    send_event('item_created', {'item': item_create.title})
    return ItemInResponse(item=ItemForResponse.from_orm(item))


@router.get("/{slug}", response_model=ItemInResponse, name="items:get-item")
async def retrieve_item_by_slug(
    item: Item = Depends(get_item_by_slug_from_path),
) -> ItemInResponse:
    return ItemInResponse(item=ItemForResponse.from_orm(item))


@router.put(
    "/{slug}",
    response_model=ItemInResponse,
    name="items:update-item",
    dependencies=[Depends(check_item_modification_permissions)],
)
async def update_item_by_slug(
    item_update: ItemInUpdate = Body(..., embed=True, alias="item"),
    current_item: Item = Depends(get_item_by_slug_from_path),
    items_repo: ItemsRepository = Depends(get_repository(ItemsRepository)),
) -> ItemInResponse:
    slug = get_slug_for_item(item_update.title) if item_update.title else None
    item = await items_repo.update_item(
        item=current_item,
        slug=slug,
        **item_update.dict(),
    )
    return ItemInResponse(item=ItemForResponse.from_orm(item))


@router.delete(
    "/{slug}",
    status_code=status.HTTP_204_NO_CONTENT,
    name="items:delete-item",
    dependencies=[Depends(check_item_modification_permissions)],
    response_class=Response,
)
async def delete_item_by_slug(
    item: Item = Depends(get_item_by_slug_from_path),
    items_repo: ItemsRepository = Depends(get_repository(ItemsRepository)),
) -> None:
    await items_repo.delete_item(item=item)
