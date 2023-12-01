import numpy as np
import matplotlib.pyplot as plt
import openai
import streamlit as st

def main():
    st.title("Fitness and Diet Tracker")

    # User Inputs
    fitness_goal = st.text_input("Enter your fitness goal (e.g., weight loss, muscle gain, endurance):")
    diet_preference = st.text_input("Enter your dietary preference (e.g., vegan, vegetarian, keto):")
    known_allergies = st.text_input("List any known food allergies (if none, enter 'None'):")

    if st.button("Generate Fitness and Diet Plan"):
        # Generating mock data
        mock_steps, mock_calories = generate_mock_fitness_data()
        caloric_balance, balance_type = calculate_caloric_balance(mock_calories, 2000)

        # Visualize data
        visualize_data(mock_steps, mock_calories)

        # Get dietary regimen from OpenAI
        dietary_regimen = get_openai_recommendations(mock_steps, caloric_balance, balance_type, fitness_goal, diet_preference, known_allergies)
        st.write("Dietary Regimen from OpenAI Assistant:")
        st.text(dietary_regimen)

        # Extract and display ingredients as a grocery list
        ingredients = extract_ingredients(dietary_regimen)
        display_grocery_list(ingredients)

def generate_mock_fitness_data():
    steps = np.random.randint(5000, 10000, size=7)
    calories = np.random.randint(1800, 2500, size=7)
    return steps, calories

def calculate_caloric_balance(calories, target_caloric_intake):
    avg_calories = np.mean(calories)
    caloric_balance = avg_calories - target_caloric_intake
    return caloric_balance, "deficit" if caloric_balance < 0 else "surplus"

def visualize_data(steps, calories):
    fig, ax = plt.subplots()
    ax.plot(steps, label='Steps')
    ax.plot(calories, label='Calories')
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_title('Weekly Steps and Caloric Intake')
    st.pyplot(fig)

def get_openai_recommendations(steps, caloric_balance, balance_type, fitness_goal, diet_preference, allergies):
    openai.api_key = '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'
    avg_steps = np.mean(steps)

    prompt = (f"Fitness goal: {fitness_goal}. Dietary preference: {diet_preference}. Allergies: {allergies}. "
              f"Average steps: {avg_steps:.0f} steps per day. "
              f"Caloric {balance_type}: {abs(caloric_balance):.0f} calories per day. "
              "Please suggest a detailed weekly dietary and exercise regimen to achieve these goals.")

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)

def extract_ingredients(dietary_regimen):
    ingredients = set()
    for line in dietary_regimen.split('\n'):
        if ':' in line:
            items = line.split(':')[1].split(',')
            for item in items:
                ingredient = item.strip().lower()
                if ingredient:
                    ingredients.add(ingredient)
    return ingredients

def display_grocery_list(ingredients):
    if ingredients:
        st.subheader("Grocery List")
        for ingredient in sorted(ingredients):
            st.write("- " + ingredient)

if __name__ == "__main__":
    main()
