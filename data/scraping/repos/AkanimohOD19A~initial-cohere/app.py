import os
import cohere
import pandas as pd
import streamlit as st

# import refresher

## Back Side
co = cohere.Client(st.secrets["COHERE_API_KEY"]) or st.secrets["COHERE_API_KEY"]


###
# we use the Generate endpoint, which generates text given the prompt.
###

@st.cache_data
## Convert Results to Dataframe - SetUp for Downloads
def convert_df(df1, df2):
    df = pd.DataFrame(list(zip(df1, df2)), columns=['IdeaName', 'Ideas'])
    return df.to_csv(index=False).encode('utf-8')


## Generate Ideas: command-nightly
def generate_idea(startup_industry, creativity, model_name):
    """
    Generate startup idea given an industry name
    Arguments:
    industry(str): the industry name
    temperature(str): the Generate model `temperature` value
    Returns:
    response(str): the startup idea
    """
    idea_prompt = f"""Generate a startup idea given the industry. Here are a few examples.

--
Industry: Workplace
Startup Idea: A platform that generates slide deck contents automatically based on a given outline

--
Industry: Home Decor
Startup Idea: An app that calculates the best position of your indoor plants for your apartment

--
Industry: Healthcare
Startup Idea: A hearing aid for the elderly that automatically adjusts its levels and with a battery \
lasting a whole week

--
Industry: Education
Startup Idea: An online primary school that lets students mix and match their own curriculum based on \
their interests and goals

--
Industry:{startup_industry}
Startup Idea: """

    # Call the Cohere Generate endpoint
    response = co.generate(
        model=model_name,
        prompt=idea_prompt,
        max_tokens=50,
        temperature=creativity,
        k=0,
        stop_sequences=["--"],
    )
    startup_idea = response.generations[0].text
    print(idea_prompt)
    print("startup_idea - pre", startup_idea)
    startup_idea = startup_idea.replace("\n\n--", "").replace("\n--", "").strip()
    print("startup_idea - post", startup_idea)
    print("-------------")
    return startup_idea


# generate_name_error = "No Error Message"

def generate_name(startup_idea, creativity, model_name):
    """
    Generate startup name given a startup idea
    Arguments:
    idea(str): the startup idea
    temperature(str): the Generate model `temperature` value
    Returns:
    response(str): the startup name
    """

    name_prompt = f"""Generate a startup name and name given the startup idea. Here are a few examples.

--
Startup Idea: A platform that generates slide deck contents automatically based on a given outline
Startup Name: Deckerize

--
Startup Idea: An app that calculates the best position of your indoor plants for your apartment
Startup Name: Planteasy

--
Startup Idea: A hearing aid for the elderly that automatically adjusts its levels and with a battery \
lasting a whole week
Startup Name: Hearspan

--
Startup Idea: An online primary school that lets students mix and match their own curriculum based on \
their interests and goals
Startup Name: Prime Age

--
Startup Idea:{startup_idea}
Startup Name:"""
    # try:
    # Call the Cohere Generate endpoint
    response = co.generate(
        model=model_name,
        prompt=name_prompt,
        max_tokens=10,
        temperature=creativity,
        k=0,
        stop_sequences=["--"],
    )
    startup_name = response.generations[0].text
    startup_name = startup_name.replace("\n\n--", "").replace("\n--", "").strip()
    # except Exception as e:
    #     print(e)
    #     generate_name_error = st.warning(f"{e}")
    return startup_name


## Front Side
st.title("🚀 Startup Idea Generator")

## Handling CSV
idea_bank = []
idea_name_bank = []

form = st.form(key="user_settings")
with form:
    industry_input = st.text_input("Industry", key="industry_input")

    col1, col2, col3 = st.columns(3)
    with col1:
        # User input - The number of ideas to generate
        num_input = st.number_input(
            "Number of ideas",
            value=3,
            step=1,
            key="num_input",
            min_value=1,
            max_value=10,
            help="Choose to generate between 1 to 10 ideas",
        )
    with col2:
        # User input - The 'temperature' value representing the level of creativity
        creativity_input = st.slider(
            "Creativity",
            value=0.5,
            key="creativity_input",
            min_value=0.1,
            max_value=0.9,
            help="Lower values generate more “predictable” output, higher values generate more “creative” output",
        )
    with col3:
        model_name = st.selectbox(
            "Please Select from the Available Models",
            ("command",
             "command-nightly",
             "command-light",
             "command-light-nightly")
        )

    # Submit button to start generating ideas
    generate_button = form.form_submit_button("Generate Idea")

    if generate_button:
        if industry_input == "":
            st.error("Industry field cannot be blank")
        else:
            # my_bar = st.progress(0.5)
            my_bar = st.snow()
            st.subheader("Startup Ideas:")

        try:
            for i in range(num_input):
                st.markdown("""---""")
                idea = generate_idea(industry_input, creativity_input, model_name)
                idea_bank.append(idea)
                name = generate_name(idea, creativity_input, model_name)
                idea_name_bank.append(name)
                st.markdown("##### " + str(name))
                st.write(idea)
                # my_bar.progress((i + 1) / num_input)
                my_bar
        except Exception as e:
            print(e)
            # refresher(60)
            st.warning(f"An Error Occured, please retry in a minute and with an Idea value less than {num_input}")


## Download A Copy
if idea_bank:
    csv = convert_df(idea_name_bank, idea_bank)

    st.download_button(
        label="Download Idea as CSV",
        data=csv,
        file_name='Idea_from_Cohere.csv',
        mime='text/csv',
    )

