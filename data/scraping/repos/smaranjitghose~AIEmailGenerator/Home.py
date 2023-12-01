import openai
import streamlit
import streamlit_lottie

from utils import *

def main():
    """
    Main Function
    """
    st.set_page_config(
        page_title="AI Audio Transciber",
        page_icon="📧",
        layout= "centered",
        initial_sidebar_state="expanded",
        menu_items={
        'Get Help': 'https://github.com/smaranjitghose/AIEmailGenerator',
        'Report a bug': "https://github.com/smaranjitghose/AIEmailGenerator/issues",
        'About': "## A minimalistic application to generate custom email templates built using Python and GPT-3"
        } )
    
    st.title("AI Email Generator")
    hide_footer()
    # Load and display animation
    anim = lottie_local("assets/animations/email.json")
    st_lottie(anim,
            speed=1,
            reverse=False,
            loop=True,
            quality="medium", # low; medium ; high
            # renderer="svg", # canvas
            height=400,
            width=400,
            key=None)

    # Initialize Session State variables(s)
    if "gen_email" not in st.session_state:
        st.session_state["gen_email"] = ""
    # User Input for Prompt
    st.markdown("#### Enter the Context of the email ⤵️")
    user_input = st.text_area(
                    label="Enter the Context of the email",
                    max_chars=1000,
                    placeholder= "Eg: Email to a professor requesting letter of recommendation for working as a Teaching Assistant in Introduction to DevOPs last semester"
                    ,label_visibility="hidden")
    if st.button("Generate Email 📜"):
        st.session_state["gen_email"] = generate_email(user_input)
        st.balloons()
        st.markdown("#### Output")
        st.markdown(st.session_state["gen_email"])


def generate_email(user_input:str):
    """
    Function to send a PUT request to the OPEN AI GPT3 API for the given prompt of generating an email
    """
    try:
        openai.api_key = st.secrets["OPENAI_KEY"] 
        model_name = "text-davinci-003"
        prompt = f"Write an professional email for the following situation:\n\nSituation: {user_input}"
        response = openai.Completion.create(
                model= model_name,
                prompt=prompt,
                temperature=0.7,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
                )
        message = response.choices[0].text
        return message
    except:
        st.error("Unable to Generate Response. Check API Key/End Point")


if __name__ == "__main__":
    main()
