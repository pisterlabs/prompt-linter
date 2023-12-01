from fastapi import Form, HTTPException, Request, APIRouter, Depends, status
import openai
import os

from database.models import Recipe
from utils import generate_prompt
from fastapi.responses import JSONResponse
from routers.auth import get_current_user, oauth2_scheme

router = APIRouter(
    prefix="/food-gpt",
    tags=["food-gpt"],
    responses={404: {"description": "Not found"}},
)

@router.post("/generate")
async def generate_recipe(request: Request, food_type: str = Form(...), ingredients: str = Form(...), cuisine: str = Form(...), token: str = Depends(oauth2_scheme)):
    user = await get_current_user(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized",
        )
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # generate prompt
    prompt = generate_prompt(food_type, ingredients, cuisine)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.6,
        max_tokens=350
    )
    # save recipe to database
    recipe_title = f"{food_type.capitalize()} {cuisine.capitalize()} recipe"
    recipes_collection = request.app.mongodb["recipes"]
    recipe = Recipe(text=response.choices[0].text, recipe_title=recipe_title, user_id=user.get("id"))
    await recipes_collection.insert_one(recipe.dict())
    return JSONResponse({"recipe": response.choices[0]})
