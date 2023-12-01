import streamlit as st
import math
import openai
import pandas as pd
import csv
import os
import plotly.express as px
#---------------------------------------FUNCTIONS------------------------------------------------------------------------------


def convert_to_kg(weight, weight_unit, height, height_unit):
    if weight_unit == 'lbs':
        weight = weight * 0.453592  # Convert lbs to kg

    if height_unit == 'inch':
        height = height * 0.0254  # Convert inches to meters
    elif height_unit == 'feet':
        height = height * 0.3048  # Convert feet to meters
    elif height_unit == 'cm':
        height = height * 0.01

    return weight, height


def calculate_bmi(weight, height):
    return weight / (height ** 2)


def calculator():
    # Define the constants and variables
    alpha = -6.322
    beta1 = -0.879  # Gender
    beta2 = 1.222   # Prescribed antihypertensive medication
    beta3 = 2.191   # Prescribed steroids
    beta4 = 0.063   # Age in years

    x1 = 0 if gender == "Male" else 1
    x2 = 1 if meds else 0
    x3 = 1 if steroids else 0

    if bmi <25:
        beta_x5 = 0
    elif 25 <= bmi < 27.5:
        beta_x5 = 0.699
    elif 27.5 <= bmi < 30:
        beta_x5 = 1.97
    elif bmi >= 30:
        beta_x5 = 2.518

    if family_history == "No diabetic 1st-Degree relative":
        beta_x6 = 0
    elif family_history == "Parent or sibling with diabetes":
        beta_x6 = 0.728
    else:
        beta_x6 = 0.753

    if smoking_history == "Non-smoker":
        beta_x7 = 0
    elif smoking_history == "Ex-smoker":
        beta_x7 = -0.218
    else:
        beta_x7 = 0.855

    exponent = alpha + (beta1 * x1) + (beta2 * x2) + (beta3 * x3) + (beta4 * age) + beta_x5 + beta_x6 + beta_x7

    probability = 1 / (1 + math.exp(-exponent))

    return probability


def chat_with_bot(message_history):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history
    )
    return response.choices[0].message['content']
#-----------------------------------------------------------------------------------------------------------------------------------------


username = None

user_login = pd.read_excel('user_login.xlsx')
username = user_login.iloc[-1, 0]
print(username)

st.title("Diabetes Risk Score")
gender = st.radio('Pick your gender',['Male','Female'])
meds = st.checkbox('Prescribed anti-hypertensive medication')
steroids = st.checkbox('Prescribed steroid')
age = st.number_input('Age', min_value=1, step=1, value=20)

family_history = st.selectbox('Family History', ['No diabetic 1st-Degree relative', 'Parent or sibling with diabetes', 'Parent and sibling with diabetes'])
col1, col2 = st.columns(2)
with col1:
    weight = st.number_input("Enter weight", min_value=1, value=60)
    height = st.number_input("Enter height", min_value=1, value=180)

with col2:
    weight_unit = st.selectbox("Unit", ['kg', 'lbs'])
    height_unit = st.selectbox("Unit", ['cm', 'inch', 'feet', 'meter'])

weight, height = convert_to_kg(weight, weight_unit, height, height_unit)
bmi = calculate_bmi(weight, height)

st.text(f"BMI Score: {bmi:.2f}")
smoking_history = st.selectbox('Smoking History', ['Non-smoker', 'Ex-smoker', 'Current smoker'])
clicked = st.button('Check Score')


if clicked:
    result = calculator()
    score = result * 100
    st.write(f"Probability of having Type-2 Diabetes : {score:.2f}%")
    st.write("Risk scores >11% are 85% sensitive for identifying diabetes (HgbA1c ≥7.0%)")
    data_to_add = [gender, meds, steroids, age, family_history, weight, height, bmi, smoking_history, score]
    with open(os.path.join("data", f"{username}_data.csv"), mode='a', newline='') as file:
        print(data_to_add)
        writer = csv.writer(file)
        # Write the data as a new row
        writer.writerow(data_to_add)



divider_placeholder = st.empty()
# Add a divider in the second part
divider_placeholder.divider()

with open("Key.txt", 'r') as file:
    api_key = file.read().strip()
openai.api_key = api_key


st.title("Chatbot with Streamlit and OpenAI")


if "message_history" not in st.session_state:
    st.session_state.message_history = [{"role": "system", "content": "YOU ARE STRICTLY A DIABETES MANAGMENT ASSISTANT WHO WILL HELP AND GUIDE USERS QUERIES."}]

if clicked:
    info = f'''Based on following user information,In 1 senetence guide user to manage or prevent diabetes.
    Gender: {gender},
    take prescribed anti-hypertensive medication?: {meds},
    take prescribed steroids?: {steroids},
    Age: {age},
    Family history: {family_history},
    Weight: {weight} in {weight_unit},
    height: {height} in {height_unit},
    BMI(Body Mass Index): {bmi},
    Smoking History: {smoking_history}'''


    user_info_message = {"role": "user", "content": info}
    st.session_state.message_history.append(user_info_message)

    response = chat_with_bot(st.session_state.message_history)
    st.session_state.message_history.append({"role": "assistant", "content": response})

#else:
#    with st.chat_message('assistant'):
#        st.markdown("How can I help you?")

for message in st.session_state.message_history[2:]:
   with st.chat_message(message["role"]):
       st.markdown(message["content"])

prompt = st.chat_input("Follow up...")
if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)

    st.session_state.message_history.append({"role":"user", "content": prompt})

    with st.chat_message("assistant"):
        print(st.session_state.message_history)
        response = chat_with_bot(st.session_state.message_history)
        print(response)
        st.markdown(response)

    st.session_state.message_history.append({"role":"assistant", "content": response})


#-----------------------------------------------------------------------------------------------------------------------------------------
divider_placeholder = st.empty()
# Add a divider in the second part
divider_placeholder.divider()

# Load your data into a Pandas DataFrame
data = pd.read_csv(f'data\\{username}_data.csv')

st.title('Pandas DataFrame Column Visualization')

# Add more spacing
st.write("")

# Visualize the selected column as an interactive histogram with Plotly
st.header(f"Visualization of Weight")
fig = px.line(data, y="weight", title=f"Distribution of Weight")
st.plotly_chart(fig)



fig = px.line(data, y="bmi", title=f"Distribution of Weight")
st.plotly_chart(fig)

# Provide additional information or descriptions
st.markdown(f"**Description:** This interactive histogram shows the distribution of BMI.")