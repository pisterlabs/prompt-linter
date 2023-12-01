import csv
import openai
from decouple import config

openai.api_key = config('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')

def load_meals_from_csv(csv_path):
    meals = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            meals.append(row)
    return meals

def generate_meal_plan(name, gender, dob, bmi, height, weight, blood_glucose_level):
    meals = load_meals_from_csv('source/recipes.csv')

    filtered_meals = []
    for meal in meals:  
        if (
            float(meal['Age']) <= float(dob) and
            float(meal['BMI']) <= float(bmi) and
            float(meal['Glucose_Level']) <= float(blood_glucose_level)
        ):
            filtered_meals.append(meal['Recipe'])

    if not filtered_meals:
        return "No matching meal plans found."

    meal_plan = f"Meal Plan for {name}:\n\n"
    meal_plan += "\n".join(filtered_meals)

    return meal_plan
