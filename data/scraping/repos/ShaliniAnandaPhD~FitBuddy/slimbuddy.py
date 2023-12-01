
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import openai

def generate_mock_fitness_data():
    steps = np.random.randint(5000, 10000, size=7)
    calories = np.random.randint(1800, 2500, size=7)
    return steps, calories

def plot_fitness_data(steps, calories):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(steps, color='blue', marker='o')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Steps', color='blue')
    ax2 = ax1.twinx()
    ax2.plot(calories, color='orange', marker='o')
    ax2.set_ylabel('Calories', color='orange')
    return fig

def get_openai_recommendations(steps, calories):
    # Use your actual API key
    openai.api_key = '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'
    avg_steps = np.mean(steps)
    avg_calories_consumed = np.mean(calories)
    target_caloric_intake = 2000

    prompt = (f"Based on an average of {avg_steps:.0f} steps per day and an average caloric intake of {avg_calories_consumed:.0f} calories per day, "
              f"and aiming for a target caloric intake of {target_caloric_intake} calories per day, "
              "can you suggest a weekly dietary regimen?")

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=300
    )

    return response.choices[0].text.strip()

# Streamlit app layout
st.title('Slim Buddy Fitness Tracker')

# Generate and display mock fitness data
st.header('Mock Fitness Data')
steps, calories = generate_mock_fitness_data()
st.write('Steps:', steps)
st.write('Calories:', calories)

# Plot and display fitness data
st.header('Fitness Data Visualization')
fig = plot_fitness_data(steps, calories)
st.pyplot(fig)

# Get and display dietary recommendations
st.header('Dietary Regimen Recommendations')
if st.button('Get Recommendations'):
    recommendations = get_openai_recommendations(steps, calories)
    st.text(recommendations)
