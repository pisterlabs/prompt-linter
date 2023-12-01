import openai
import streamlit as st
from ml_backend import ml_backend

st.title("Automated Email Generator App")

coder = '<h1 style="font-family:Times-New-Roman; color:Red ; font-size: 20px;">by Hetvi Bhora</h1>'
st.markdown(coder,unsafe_allow_html=True)

st.title("Generate Email")

backend = ml_backend()

with st.form(key="form"):
    prompt = st.text_input("Describe the Kind of Email you want to be written.")
    st.text(f"(Example: Write me a professional sounding email to my boss)")

    start = st.text_input("Begin writing the first few or several words of your email:")

    slider = st.slider("How many characters do you want your email to be? ", min_value=64, max_value=750)
    st.text("(A typical email is usually 100-500 characters)")

    submit_button = st.form_submit_button(label='Generate Email')

    if submit_button:
        with st.spinner("Generating Email..."):
            output = backend.generate_email(prompt, start)
        st.markdown("# Email Output:")
        st.subheader(start + output)

        st.markdown("____")

        st.subheader("You can press the Generate Email Button again if you're unhappy with the model's output")

        st.markdown("____")

        st.markdown("# Send Your Email Now")

        
        # st.subheader("Otherwise:")
        # st.text(start + output)
        url = "https://mail.google.com/mail/?view=cm&fs=1&to=&su=&body=" + backend.replace_spaces_with_pluses(start + output)

        st.markdown("[Click me to send the email]({})".format(url))
