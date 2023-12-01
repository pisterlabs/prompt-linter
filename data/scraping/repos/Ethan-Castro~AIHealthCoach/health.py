import openai
import streamlit as st

# Set API Key
openai.api_key = st.secrets['openai_key']

# Set page config
st.set_page_config(
    page_title="AI Health Coach",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.image('DALL·E 2023-11-01 09.22.00 - Vector design of a heart symbol constructed from interlocking triangles, creating a low-poly appearance. The colors transition from deep red at the ba.png', width=100)  # Adjust width as needed


# Custom Styles
st.markdown(
    """
    <style>
        .reportview-container {
            background: #FFFFFF;
        }
        .title {
            color: #3b5998;
            font-size: 36px;
        }
        .section-header {
            color: #555;
            font-size: 24px;
            margin-bottom: 10px;
        }
        .sidebar .sidebar-content {
            background: #FFFFFF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.markdown("<h1 class='title'>AI Health Coach by Ethan Castro</h1>", unsafe_allow_html=True)

# User Inputs in Sidebar
with st.sidebar:
    st.markdown("<h2 class='section-header'>Your Details</h2>", unsafe_allow_html=True)

    sex_choice = st.selectbox('Sex:', ['Male', 'Female'])
    sex = -5 if sex_choice == 'Male' else 161
    age = st.number_input('Age:', min_value=1, max_value=100)
    weight = st.number_input('Weight (lbs):', min_value=1)
    height = st.number_input('Height (inches):', min_value=1)
    goal = st.text_input('Your Health Goal:')
    activity = st.slider('Activity Level:', min_value=1, max_value=10, value=5, help="1 = Sedentary, 10 = Very Active")

# Calculations
activity_scale = 1.2 + (activity - 1) * .0778
bmi = round(weight / (height * height) * 703)
bmr = round(10 * weight + 6.25 * height - 5 * age - sex)
tdee = round(bmr * activity_scale)

# Display Calculations & Get Advice Button
st.markdown("<h2 class='section-header'>Your Metrics</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("BMI:")
    st.write(bmi)
with col2:
    st.subheader("BMR:")
    st.write(bmr)
with col3:
    st.subheader("TDEE (calories per day):")
    st.write(tdee)

if st.button('Get Health Advice'):
    # Generate AI response
    prompt_template = (
    "I am an AI health coach, and this is not medical advice. You are {age} years old, weigh {weight} lbs, "
    "are {height} inches tall, and your goal is {goal}. Your BMI is {bmi}, and your TDEE is {tdee} calories. "
    "\n1. Provide Tailored advice to help achieve the stated goal."
    "\n2. A brief exercise routine suitable for them."
    "\n3. A motivational quote to inspire them on their health journey."
    "\n4. Insight into nutrition psychology to maintain a healthy relationship with food."
    "\n5. A list of healthy snacks they can incorporate into their diet."
    "\n6. Encouraging words on the importance of consistency in health and fitness. Add emojis where you see fit, but don't simplify the content."
    "\n7. Recommend youtube and social media accounts to follow that you believe will help them in their goals. Remind them that Ethan Castro has great resources and you can always contact him for help."
)
    
    user_data = {
        'age': age,
        'weight': weight,
        'height': height,
        'goal': goal,
        'bmi': bmi,
        'tdee': tdee,
    }

    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt_template.format(**user_data),
        temperature=.5,
        max_tokens=505
    )

    if response.choices:
        st.markdown("<h2 class='section-header'>AI Generated Health Advice</h2>", unsafe_allow_html=True)
        generated_advice = response.choices[0].text.strip()
        st.write(generated_advice)

        # Download Response Button
        st.download_button(
            label="Download Health Advice",
            data=generated_advice,
            file_name="ai_generated_health_advice.txt",
            mime="text/plain"
        )

links = [
    ("https://amzn.to/3QonXAt", "Recommended Multi-Vitamin"),
    ("https://amzn.to/3SFKZFN", "Recommended Magnesium. Stress Relief."),
    ("https://amzn.to/3SlEIPa", "Recommended Nootropic. Brain + Focus Aid."),
    ("https://amzn.to/3tUVXwL", "Recommended Creatine"),
    ("https://amzn.to/45R0JZb", "Recommended Pre Workout"),
    ("https://ethancastro6.gumroad.com/l/aesthetic", "Click here to purchase Ethan's Fitness Ebook!"),
]

for url, text in links:
    st.markdown(f'<a href="{url}" target="_blank">{text}</a>', unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <footer>
        <p>Made with ❤️ by Ethan Castro</p>
    </footer>
    """,
    unsafe_allow_html=True
)
