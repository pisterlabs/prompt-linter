import streamlit as st
import openai

# Set OpenAI key
openai.api_key = 'YOUR API KEY'

# Define the function to generate text
def generate_text(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',  # Use the GPT-4 model
        prompt=prompt,
        max_tokens=3800  # Adjust as needed
    )
    return response.choices[0].text.strip()

st.title("SwymGPT")


st.write("Welcome to SwymGPT: Your AI-Powered Swimming Training Program Generator")

st.write("SwymGPT uses OpenAI's Large Language Model (LLM) technology to generate customized swimming training programs. Although it's not verified by sports scientists or professional swimming coaches, it offers a fascinating example of AI and LLM's potential application in the sports industry.")

st.write("Feel free to experiment with the application by varying inputs to generate unique training programs. It offers a refreshing approach to swimming training and provides insights into the transformative potential of AI.")

st.write("Enjoy your swim!")

st.write("Know more about the developer: https://www.linkedin.com/in/acecanacan/")

# Get user input and generate prompt
swimming_event = st.selectbox("Select the swimming event type:", 
                              ["Freestyle", "Backstroke", "Breaststroke", "Butterfly", 
                               "Medley", "Freestyle Relay", "Medley Relay", "Open Water"])
distance_options = {
    "Freestyle": ["50m", "100m", "200m", "400m", "800m", "1500m"],
    "Backstroke": ["100m", "200m"],
    "Breaststroke": ["100m", "200m"],
    "Butterfly": ["100m", "200m"],
    "Medley": ["200m", "400m"],
    "Freestyle Relay": ["4 x 100m", "4 x 200m"],
    "Medley Relay": ["4 x 100m (Men)", "4 x 100m (Women)", "4 x 100m (Mixed)"],
    "Open Water": ["10km"]
}
distances = distance_options.get(swimming_event, [])
selected_distance = st.selectbox("Select the distance:", distances)
weeks_before_event = st.number_input("How many weeks before the event?", min_value=1)
equipment_options = ["No equipment", "Pool buoy", "Fins", "Kickboard", "Paddles", "Pull rope", "Swim snorkel"]
selected_equipment = st.multiselect("Select your equipment:", equipment_options, default="No equipment")
fitness_level = st.slider("Your fitness level (1-10):", min_value=1, max_value=10)
weight = st.number_input("Your weight in kilograms:", min_value=1.0)
height = st.number_input("Your height in cm:", min_value=1.0)
age = st.number_input("Your age:", min_value=1)
swimming_pace = st.number_input("Your average swimming pace for 100 meters (in seconds):", min_value=1.0)
experience_years = st.number_input("Your total years of swimming training:", min_value=0)
workout_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_workout_days = st.multiselect("Which days can you workout?", workout_days)
workout_duration = st.slider("Your workout duration in minutes:", min_value=0, max_value=180)
additional_note = st.text_input("Additional Note if Any")

if swimming_event and selected_distance:
    st.write(f"You have chosen {selected_distance} {swimming_event.lower()}.")

# Generate the input summary
if st.button("Generate"):
    with st.spinner("Generating..."):
        prompt = f"""
        Please provide a simple and structured training plan based on the following information:
        Make sure that the format can easily be understood
        The intervals should be in meters not minutes

        Make a  {weeks_before_event} week swimming training plan
        Age: {age}
        Weight: {weight}kg
        Height: {height}cm
        Fitness level: {fitness_level}/10
        Swimming experience: {experience_years} years
        Event: {selected_distance} {swimming_event.lower()}
        Equipment: {', '.join(selected_equipment)}
        Average swimming pace: {swimming_pace} seconds per 100m
        Workout days: {', '.join(selected_workout_days)}
        Workout duration: {workout_duration} minutes each session
        Training duration: {weeks_before_event} weeks
        {additional_note}

        Do not put additional notes.
        All training plans per week should be included

        """
        training_plan = generate_text(prompt)
        st.write(training_plan)

# Run the app
if __name__ == '__main__':
    st.write("")
