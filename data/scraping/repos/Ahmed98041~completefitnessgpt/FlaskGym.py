from flask import Flask, request, jsonify
from flask_cors import CORS  # Importing the CORS library
import openai
import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv('key.env')

# Set up your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__, static_folder='static')

# Enable CORS for all origins
CORS(app)

def fitness_evaluation(age, height_meters, weight_kg, gender, activity_levels_1highest, free_time_daily_hours, a_pwd, additional_info, user_goals, target_muscles, available_days, existing_workout_plan):
    messages = []
    messages.append({
        "role": "system",
        "content": "You are a specialized fitness AI tasked with generating detailed and personalized fitness and diet plans for individuals based on a variety of input parameters. Your task is to create a tailored plan that takes into consideration the user's age, height, weight, gender, activity level, daily free time, personal goals, target muscles, available days for workouts, and existing workout plans. Please divide the output into two sections: a fitness plan and a diet plan. The fitness plan should be presented in a grid format where each cell represents a day and outlines the workouts targeting the specified muscles for that day. The diet plan should complement the fitness plan and be structured according to different meals throughout the day. Make sure to craft plans that suit a wide range of demographics."
    })
    pwd = "a pwd" if a_pwd else "not a pwd"
    additional_info = additional_info if additional_info else "nothing"

    message = f"Give a person with age of {age}, height of {height_meters} meters, weight of {weight_kg} kg, activity levels of {activity_levels_1highest} with 1 being the highest activity level, free time daily of {free_time_daily_hours} in hours, is {pwd}, and has additional info of {additional_info}. User goals: {user_goals}. Target muscles: {', '.join(target_muscles)}. Available days: {', '.join(available_days)}. Existing workout plan: {existing_workout_plan}"
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
    temperature=0.7,
)
    reply = response["choices"][0]["message"]["content"]
    fitness_advice = [{"title": "Fitness Advice", "content": reply}]
    
    return fitness_advice

@app.route("/fitness_advice", methods=["POST"])
def get_fitness_advice():
    # Get user inputs from the JSON body
    data = request.get_json()
    age = data["age"]
    height_meters = float(data["height_meters"])
    weight_kg = int(data["weight_kg"])
    gender = data["gender"]
    activity_levels_1highest = data["activity_levels_1highest"]
    free_time_daily_hours = float(data["free_time_daily_hours"])
    a_pwd = data["a_pwd"]
    additional_info = data["additional_info"]
    user_goals = data["user_goals"]
    target_muscles = data["target_muscles"]
    available_days = data["available_days"]
    existing_workout_plan = data["existing_workout_plan"]

    # Call the fitness_evaluation function and get the results
    fitness_advice = fitness_evaluation(age, height_meters, weight_kg, gender, activity_levels_1highest, free_time_daily_hours, a_pwd, additional_info, user_goals, target_muscles, available_days, existing_workout_plan)

    return jsonify(fitness_advice=fitness_advice)


if __name__ == "__main__":
    app.run(debug=True)
