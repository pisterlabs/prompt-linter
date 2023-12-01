from openai_function_call import OpenAISchema
from pydantic import Field, BaseModel
from typing import List

class Ingredient(BaseModel):
    """Ingredient is a representation of one of the ingredients in the recipe."""
    name: str = Field(..., description="Ingredient name")
    unit: str = Field(..., description="abbreviated unit name in standard form, empty if none specified")
    quantity: str = Field(..., description="quantity of ingredients, empty if none specified")

class RecipeDetails(OpenAISchema):
    """RecipeDetails are the details describing a recipe. Details are precise and
reflect the input from the user."""
    description: str = Field(..., description="casual description of every recipe step in approximately 200 characters")
    steps: List[str] = Field(..., description="""specific list of all steps in the recipe, in order.
If multiple steps can be combined into one, they will.""")
    recipe_title: str = Field(..., description="title of the recipe in 20 characters or less")
    ingredients: List[Ingredient] = Field(..., description="list of all ingredients in the recipe")
    total_time: int = Field(..., description="total time in minutes required for this recipe")
    active_time: int = Field(..., description="total active (non-waiting) cooking time in minutes required for this recipe")
