"""Functionality to determine risk factors for users"""
import torch
import cohere
from model.model import DiabetesPredictorModel
from keys import COHERE_API_KEY # Secret

# Load our custom predictive health model
model = DiabetesPredictorModel()
model.load_state_dict(torch.load("./model/model.pt", map_location=torch.device("cpu")))
model.eval()

# Access the Cohere API for GPT powered text generation
co = cohere.Client(COHERE_API_KEY)

# Measure how far user deviates from ideal metrics
ideal_health = { 
    "high blood pressure": 0.0,             # 0 = No high BP, 1 = high BP
    "high cholesterol": 0.0,                # 0 = No high cholesterol, 1 = high cholesterol
    "cholesterol check": 0.0,               # 0 = No cholesterol check in last 5y, 1 = cholesterol check in last 5 years (Discard)
    "bmi": 25.0,                            # Body mass index (weight/height^2)
    "smoker": 0.0,                          # 0 = Non smoker, 1 = smoker
    "history of stroke": 0.0,               # 0 = No history of stroke, 1 = history of stroke
    "history of heart issues": 0.0,         # 0 = No history of heart disease or heart attacks, 1 = history of heart disease/attack
    "lack of physical activity": 1.0,       # 0 = No physical activity in the past 30 days, 1 = physical activity in past 30 days
    "inadequate fruit intake": 1.0,         # 0 = Does not consume 1 or more fruits per day, 1 = consumes fruit 1 or more times per day
    "inadequate vegetable intake": 1.0,     # 0 = Does not consume 1 or more veggies per day, 1 = consumes 1 or more veggies per day
    "excess alcohol consumption": 0.0,      # 0 = No heavy alcohol consumption, 1 = heavy alcohol consumption (14+ drinks/week for men, 7+ drinks/week for women)
    "access to healthcare": 1.0,            # 0 = No healthcare plan, 1 = has a healthcare plan
    "general health": 1.0,                  # 1 = Excellent health, 2 = very good health, 3 = good health, 4 = fair health, 5 = poor health
    "mental health": 0.0,                   # How many of the past 30 days has a mental health issue affected you?
    # (Discard sex/age, as those are not risk factors the user can alter)
}

def get_risk_factors(data):
    """Generate a rundown of risk factors given their 16 point input"""
    risk_factors = []
    for i, (key, value) in enumerate(ideal_health.items()):
        if key == "bmi":
            if data[i] > value:
                if data[i] > value + 5:
                    risk_factors.append("obese bmi")
                else:
                    risk_factors.append("overweight bmi")
        elif key == "general health":
            if data[i] >= 4:
                risk_factors.append("poor general health")
        elif key == "mental health":
            if data[i] >= 7:
                risk_factors.append("struggles with mental health")
        elif key == "cholesterol check":
            pass # Not an important factor
        elif data[i] != value:
            risk_factors.append(key)
    return risk_factors

def generate_prediction(data):
    """Generate a prediction of diabetes risk based on the 16 factors"""
    data = torch.tensor([data])
    with torch.no_grad():
        return model(data).item()

def generate(data):
    """Based on listed risk factors and likelihood of diabetes/pre diabetes, generate a paragraph describing personalized health improvements"""
    diabetes_percent = 100 * generate_prediction(data)
    likely_pre_diabetic = False
    if diabetes_percent > 70:
        likely_pre_diabetic = True
    risk_factors = ", ".join(get_risk_factors(data))
    if likely_pre_diabetic:
        prompt = f"Inform a user about how they could improve their lifestyle to reduce their likelihood of diabetes developing. Their lifestyle carries a {diabetes_percent} percent chance of developing diabetes, and they are likely pre diabetic. There risk factors are: {risk_factors}. Use a positive but professional tone."
    else:
        prompt = f"Inform a user about how they could improve their lifestyle to reduce their chance of getting diabetes or pre diabetes. Their lifestyle carries a {diabetes_percent} percent chance of developing diabetes or pre diabetes. There risk factors are: {risk_factors}. Use a positive but professional tone."
    response = co.generate(
        prompt=prompt,
        max_tokens=500,
        num_generations=1,
    )
    return diabetes_percent, response
