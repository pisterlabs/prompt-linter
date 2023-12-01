# this is a chatbot to talk about metabolic health

# libraries
import openai # for AI
import streamlit as st # visual interface

# need api key when running locally

# acttivates openAI with the user prompt + predefined expertise as a metabolic health expert
def generate_response(myprompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Your role is to act like a nutrition and metabolic health expert. Give your answers accordingly.###" + myprompt,
        temperature=.3,
        max_tokens=1024
    )
    return response.choices[0].text.strip()

# gathers user input
def main ():
    st.title("At home health coach") # title in the visual interface

    # form to be submitted
    with st.form("Basic information required"):
        st.write("Basic information required")
        age = st.slider("Approximate age")
        bmi = st.slider("Approximate BMI")
        gender = st.text_input('Your gender', key = 'Your gender')
        ques = st.text_input('Your question', key = 'Your question')
        checkbox_val = st.checkbox("Form checkbox") 
        submitted = st.form_submit_button("Submit")

    if submitted:
       # join the above information in a single variable to be sent to AI
       myprompt = "I am %d years old %s, my BMI is %d. My question is %s" % (age, gender, bmi, ques)
       st.write(generate_response(myprompt))

if __name__ == "__main__":
    main()