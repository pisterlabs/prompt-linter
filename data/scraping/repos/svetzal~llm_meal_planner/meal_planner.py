from langchain import PromptTemplate, LLMChain
from langchain.output_parsers import PydanticOutputParser

from meal_plan import MealPlan


class MealPlanner:
    def __init__(self, household, llm):
        self.household = household
        self.llm = llm
        prompt = """

You are an expert meal planner who really cares about people's happiness, health and nutrition. You must not ever 
include foods to which your people are allergic. Try to limit the use of foods they dislike. Try to include their 
favourite foods as much as possible. The house only has a limited number of appliances and cookware, so you need to 
make sure that you don't plan meals that require different appliances or cookware than they have. Try to re-use 
ingredients between meals and snacks as much as possible to reduce waste. Meals should decrease in calories 
throughout the day.

Food Allergies (never include food that will trigger these): {allergies}
Available appliances: {available_appliances}    
Available cookware: {available_cookware}
Favourite foods: {favourite_foods}
Disliked foods: {disliked_foods}

Respond in the following format:
{format_instructions}

Create a meal plan for a household of {family_size} that includes breakfast, lunch, dinner, and snacks for {days} days.

    """

        self.parser = PydanticOutputParser(pydantic_object=MealPlan)
        task = PromptTemplate(
            input_variables=["days"],
            template=prompt.strip(),
            partial_variables={
                "allergies": ", ".join(self.household.food_allergies),
                "available_appliances": ", ".join(self.household.equipment.appliances),
                "available_cookware": ", ".join(self.household.equipment.cookware),
                "favourite_foods": ", ".join(self.household.food_preferences.likes),
                "disliked_foods": ", ".join(self.household.food_preferences.dislikes),
                "family_size": self.household.size,
                "format_instructions": self.parser.get_format_instructions(),
            }
        )
        self.chain = LLMChain(llm=self.llm, prompt=task, verbose=True)

    def plan_days(self, days):
        response = self.chain.run(
            output_parser=self.parser,
            days=days,
        )
        return response
