from api.google_api_calendar_client import GoogleCalendarClient
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser

from models.day import Day
from models.meal import Meal
from models.prompt_template import RecipePromptTemplate
from models.recipe import Recipe
from models.recipe_assistant import RecipeAssistant
from models.recommendation import Recommendation
from models.user_settings import UserSettings
from models.week import Week
from utils.constants import *
from utils.database_handler import DatabaseHandler
from utils.loggerX import Logger


logger = Logger(__name__)

class food_ai():
    def __init__(self):
        self.recommendation_model = OpenAI(model_name=AI_MODEL_GPT_4, temperature=0.9, max_tokens=1024)
        self.recipe_model = OpenAI(model_name=AI_MODEL_GPT_35_TURBO, temperature=0.2, max_tokens=1024)
        self.db_handler = DatabaseHandler()
        self.user_settings = None
        
    def update_user_settings(self, skill_level, dietary_restrictions,
                             preferences, budget_time_period, budget_amount,
                             num_servings, meal_type):

        self.user_settings = UserSettings(
            skill_level=skill_level,
            dietary_restrictions=dietary_restrictions,
            preferences=preferences,
            budget_time_period=budget_time_period,
            budget_amount=budget_amount,
            num_servings=num_servings,
            meal_type=meal_type,
        )
        
    def set_day_settings(self, day):
        self.user_settings.day = day
        
    def get_recommendation(self):
        if self.user_settings is None:
            raise Exception("User settings not set")

        logger.info_header_footer("user_settings", self.user_settings.__str__())

        # Set up parsers
        recipe_parser = PydanticOutputParser(pydantic_object=Recipe)
        recommendation_parser = PydanticOutputParser(pydantic_object=Recommendation)

        # Set up prompts
        recommendation_prompt = RecipePromptTemplate(recommendation_parser)
        recipe_prompt = RecipePromptTemplate(recipe_parser)

        # Set up assistant
        assistant = RecipeAssistant(self.user_settings,
                                    self.recommendation_model,
                                    self.recipe_model,
                                    recommendation_parser,
                                    recipe_parser,
                                    recommendation_prompt,
                                    recipe_prompt)

        # Get recommendation
        recommendation = assistant.get_recommendation(self.user_settings.day)
        logger.info_header_footer("recommendation", recommendation)

        # Get recipe
        recipe = assistant.get_recipe(recommendation)
        logger.info_header_footer("recipe", recipe)

        return recommendation, recipe

    def save_recipe(self, recipe):
        # Create meal, day, and week objects
        meal = Meal(recipe.recipe_title,
                    recipe.ingredients,
                    recipe.prep_steps,
                    recipe.cook_time,
                    recipe.day,
                    db_handler=self.db_handler)
        day = Day(self.user_settings.day, db_handler=self.db_handler)
        week = Week(db_handler=self.db_handler)

        # Add meal to day and day to week
        day.add_meal(meal)
        week.tue = day
        week.set_week_dates()

        #TODO save to database
        return meal, week

    def create_calendar_event(self, meal, week):
        # Save meal to calendar
        client = GoogleCalendarClient(GOOGLE_CREDENTIALS_FILE, GOOGLE_TOKEN_FILE, scopes=["https://www.googleapis.com/auth/calendar"])
        client.create_meal_event(meal, week.start_date)
