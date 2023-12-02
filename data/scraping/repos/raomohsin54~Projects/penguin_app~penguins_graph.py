import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# Set page configuration
st.set_page_config(
    page_title="Palmer's Penguins",
    page_icon=":penguin:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define styles for the app
styles = """
<style>
img {
    max-width: 100%;
}
.sidebar .sidebar-content {
    background-color: #f5f5f5;
}
</style>
"""

# Render styles
st.markdown(styles, unsafe_allow_html=True)

image = Image.open(r"C:\Users\mmukhtiar\Downloads\emperorpairwchick_Colin-Mcnulty_CROP_Web.jpg")

# Define header
header = st.container()
with header:
    st.image(image)
    st.title("Palmer's Penguins")
    st.markdown("Use this Streamlit app to make your own scatterplot about penguins!")
    st.write("")

# Define main content
content = st.container()
with content:
    # Load penguins dataset
    penguin_file = st.file_uploader('Select Your Local Penguins CSV (default provided)')
    if penguin_file is not None:
        penguins_df = pd.read_csv(penguin_file)
    else:
        st.warning("Please select a CSV file to continue.")
        st.stop()

    # Select x and y variables
    st.subheader("Create a scatterplot")
    st.write("Select the x and y variables to create a scatterplot.")
    col1, col2 = st.beta_columns(2)
    with col1:
        selected_x_var = st.selectbox('X variable', ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
    with col2:
        selected_y_var = st.selectbox('Y variable', ['bill_depth_mm', 'bill_length_mm', 'flipper_length_mm', 'body_mass_g'])

    # Create scatterplot
    fig, ax = plt.subplots()
    ax = sns.scatterplot(x = penguins_df[selected_x_var], y = penguins_df[selected_y_var], hue = penguins_df['species'])
    plt.xlabel(selected_x_var)
    plt.ylabel(selected_y_var)
    plt.title("Scatterplot of Palmer's Penguins")
    st.pyplot(fig)

# Define sidebar
sidebar = st.sidebar
with sidebar:
    st.image(r"C:\Users\mmukhtiar\Downloads\cornelius-ventures-Ak81Vc-kCf4-unsplash.png")
    st.subheader("Get insights about the data")
    st.write("Enter a prompt to generate insights about the data using PandasAI and OpenAI.")
    prompt = st.text_input("Enter your prompt:")
    if prompt:
        # Initialize PandasAI and OpenAI
        llm = OpenAI()
        pandas_ai = PandasAI(llm)
        
        # Run PandasAI with user input prompt
        result = pandas_ai.run(penguins_df, prompt=prompt)
        
        # Display result
        if result is not None:
            st.write("### Insights")
            st.write(result)
