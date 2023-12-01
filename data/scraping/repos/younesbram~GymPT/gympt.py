import streamlit as st
import openai
import time

st.set_page_config(
    page_title="GymPT",
    page_icon="💪",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": "https://www.x.com/didntdrinkwater/",
        "Report a bug": "https://www.younes.ca/contact",
        "About": "# AI Comedy\nAn app that uses NLP to generate hilarious skits!",
    },
)
# Sidebar for OpenAI API Key Input
st.sidebar.title('Insert OpenAI API Key to use GPT4')
user_openai_key = st.sidebar.text_input('Enter OpenAI API Key (Please):')

# Use the user-provided key if available, otherwise use the secret key
openai_api_key = user_openai_key if user_openai_key else st.secrets["OPENAI_API_KEY"]

# Set the OpenAI API key
openai.api_key = openai_api_key

# Replace with your OpenAI API key
# openai.api_key = ""

# Function to calculate TDEE
def calculate_tdee(height, weight, activity_level, goal, age, units):
    # Convert height and weight to centimeters and kilograms
    if units == 'inches/lbs':
        height_cm = height * 2.54
        weight_kg = weight * 0.453592
    else: # Assuming the other option is 'cm/kg'
        height_cm = height
        weight_kg = weight

    # Calculate BMR using Mifflin-St Jeor Equation
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5

    # Multiply by activity factor
    tdee = bmr * activity_level

    # Adjust for goal (e.g., weight loss)
    #if goal == "Weight Loss":
    #    tdee -= 350  # Example deficit for weight loss
    #elif goal == "Muscle Gain":
   #     tdee += 350
   #currently gpt4 already does this part 
    return tdee
    

# Title
st.title('💪 GymPT: Personalized Fitness')

# Introduction
st.markdown("""
Welcome to GymPT, your personal guide to achieving the body of your dreams! 
Rome wasn't built in a day, & neither were the bodies of legends Arnold Schwarzenegger, Mike Mentzer, Jay Cutler, and more. 
We believe in the power of consistency, dedication, and intelligent training. 
Whether you're aiming for Herculean strength, chiseled aesthetics, or optimal health, 
GymPT is designed to support you every step of the way. 
Remember, the road to success is not about quick fixes—it's about hard work, 
smart choices, and never giving up. Now, let's build your personalized plan! (If you have an OpenAI API key, please open the sidebar and insert it. For now, this app is free to use thanks to x.com/didntdrinkwater)
""")

# User Input for Workout Goals
goal = st.selectbox('Choose Your Fitness Goal', ['Weight Loss', 'Muscle Gain', 'Maintenance'])

# User Input for Dietary Preferences
diet = st.multiselect('Select Dietary Preferences (Optional)', ['Vegan', 'Keto', 'Low-Carb', 'High-Carb', 'Carb-Cycling', 'Gluten-Free'])

# Items in Fridge (for personalized diet recommendations)
fridge_items = st.text_area('Items in Your Fridge (Optional, leave empty if you only want a workout regimen)', value='', placeholder='E.g., eggs, chicken, broccoli, almonds...')


# Preferred Training Styles
training_styles = st.multiselect('Select Your Preferred Training Style - You can mix and match up to 3 trainers thanks to AI (Optional)', [
'Arnold Schwarzenegger – Volume Training and Classic Physique',
'Mike Mentzer – High-Intensity Training (HIT)',
'Jay Cutler – Balanced Approach with Emphasis on Symmetry',
'Dorian Yates – HIT with Blood and Guts Training',
'Frank Zane – Focus on Proportion and Aesthetics',
'Ronnie Coleman – High Volume and Heavy Lifting',
'Lee Haney – Stimulate, Don\'t Annihilate; Emphasis on Recovery',
'Calisthenics – Bodyweight Training for Strength and Flexibility',
'Rich Gaspari – Pre-Exhaustion Training with Intensity',
'Lou Ferrigno – Power Bodybuilding with Heavy Weights',
'Sergio Oliva – Classic Mass Building with Frequent Training',
'Larry Scott – Focus on Arms and Shoulders',
'Tom Platz – High Volume Leg Specialization',
'Flex Wheeler – Quality over Quantity; Focus on Form',
'Phil Heath – Scientific Approach with Attention to Detail',
'Chris Bumstead – Classic Physique with Modern Training',
'Kai Greene – Mind-Muscle Connection and Artistic Expression',
'CrossFit – Functional Fitness with Varied High-Intensity Workouts',
'Powerlifting – Focus on Strength and Power',
'Yoga – Focus on Flexibility and Mindfulness',
'Pilates – Focus on Core Strength and Posture',
'HIIT – High-Intensity Interval Training',
'Fasted Cardio – Cardio on an Empty Stomach',
'Kickboxing – Martial Arts and Cardio',
'Boxing – Martial Arts and Cardio',
'Muay Thai – Martial Arts and Cardio',
'Karate – Martial Arts',
'Taekwondo – Martial Arts',
'Zumba – Dance Fitness',


], max_selections=3)


# Height and Weight Inputs
units = st.selectbox('Choose Your Units', ['inches/lbs', 'cm/kg'])

if units == 'inches/lbs':
    height_description = 'Enter Your Height (e.g., 68 inches)'
    weight_description = 'Enter Your Weight (e.g., 160 lbs)'
else: # Assuming the other option is 'cm/kg'
    height_description = 'Enter Your Height (e.g., 172 cm)'
    weight_description = 'Enter Your Weight (e.g., 73 kg)'

height = st.number_input(height_description, min_value=0, max_value=300, step=1)
weight = st.number_input(weight_description, min_value=0, max_value=500, step=1)

age = st.number_input('Enter Your Age', min_value=0, max_value=120, step=1)

# Activity Level
activity_levels = {
    "Sedentary (little to no exercise)": 1.2,
    "Lightly active (light exercise/sports 1-3 days/week)": 1.375,
    "Moderately active (moderate exercise/sports 3-5 days/week)": 1.55,
    "Very active (hard exercise/sports 6-7 days a week)": 1.725,
    "Super active (very hard exercise/sports & physical job or training twice a day)": 1.9
}
activity_level = st.selectbox('Choose Your Activity Level', list(activity_levels.keys()))
activity_factor = activity_levels[activity_level]

def generate_plan(goal, diet, fridge_items, training_styles, tdee, age):
    messages = [
        {
            "role": "system",
            "content": f"You are an extremely detailed Ai, who is knowledgeable in bodybuilding/fitness/dietitian and an expert! You only respond ethically."
        },
        {
            "role": "user",
            "content": f"My dietary preferences are {diet}. Create the perfect curated plan from {training_styles}. If there is anything in my fridge {fridge_items}, please include a meal plan, if not, dont mention the fridge being empty. My TDEE is {tdee} and I am {age} years old. My fitness goal is {goal} so try to give me accurate response based off my info. If i withheld dietary preference or training style, IGNORE IT and carry on with generic response. Do not give me any extra info, just respond as the trainers or mix of trainers and give the workout plan and the philosophy along with some things to research if need be and quotes from the trainers if there are any. Be extremely detailed and straight to the point"
        }
    ]

    delay_time = 0.01
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.666666666666666666666666666666420,
        stream=True,
    )

    # Container to incrementally update the display
    c = st.empty()

    generated_text = ''
    for event in response:
        event_text = event['choices'][0]['delta'].get('content', '')
        generated_text += event_text
        c.markdown(generated_text) # Update the entire accumulated text
        time.sleep(delay_time)

    return generated_text




# Generate Workout and Diet Plan
if st.button('Generate Plan'):
    # Validation checks
    if not height or not weight or not age or not activity_level:
        st.error('Please fill in all required fields (Height, Weight, Age, and Activity Level) before generating the plan.')
    else:
        with st.spinner('We\'re all gonna make it brah... Generating...'):
            # Calculate TDEE
            tdee = calculate_tdee(height, weight, activity_levels[activity_level], goal, age, units)

            # Check if TDEE is calculated
            if tdee:
                # Call the generate_plan function with the calculated TDEE
                plan = generate_plan(goal, diet, fridge_items, training_styles, tdee, age)
                # Check if the plan has been generated
                if plan:
                    # Create a download button for the generated plan
                    st.download_button(
                    label="Download Your Plan",
                    data=plan,
                    file_name="generated_plan.txt",
                    mime="text/plain",
    )

            else:
                st.error('An error occurred while calculating your plan. Please make sure all inputs are correct.')
