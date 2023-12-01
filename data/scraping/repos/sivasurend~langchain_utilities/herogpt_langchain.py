
import streamlit as st
import langchain
from langchain.llms import OpenAI
from langchain import PromptTemplate
import requests


# Your Python script
def run_script(name, bad_alternative, better_solution, objections, icp, creativity):
    llm = OpenAI(model_name="text-davinci-003", temperature=creativity, openai_api_key=openai_api_key)
    template = """
              I want you to assume the role of the marketing manager of a startup and you've been assigned to come up 
              with a fantastic H1 header message for the hero section of the website.

              You will accept a five parameters of input in order to get more context about the business and come up 
              with stunning hero H1 header message for the startup.

              The input fields are,

              name of the startup = {name}
              What bad alternative do people resort to when they lack your product? = {bad_alternative}
              How is your product better than that bad alternative? = {better_solution}
              What objections might the user have to use your product? = {objections}
              Ideal customer profile = {icp}

              The output should be 3 awesome hero header message options for the startup's website inspired by below examples.

              below or some of the best examples of a fantastic hero header message. use these to train yourself.

              Name of the Startup: Airbnb
              Bad Alternatives: Stuck in sterile hotels, don't experience the         real culture
              Objections: Only available for long-term rentals
              Your startup’s better solution: Stay in locals' homes.
              Action Statement: Experience new cities like a local.
                    Header: Experience new cities like a local in rentals. No         minimum stays.

              Name of the Startup: Dropbox
              Bad Alternatives: Unorganized paper files,easily lost flashdrives
              Objections: Risk of low-privacy
              Your startup’s better solution: Online cloud storage that               automatically syncs the cloud your files
              Action Statement:: Upload your files to the cloud automatically.
              Header: Upload your files to the cloud automatically. Chosen by         over half of the Fortune 500s for our superior security.

              Name of the Startup: Doordash
              Bad Alternatives: Long waits at restaurants and traffic-heavy           trips to get food
              Objections: High delivery costs
              Your startup’s better solution: Quick deliveries from local             restaurants.
              Action statement: Get your favorite meals with the press of a         button
              Header: Get your favorite meals with the press of a button. No         extra fees.

              Name of the Startup: Webflow
              Bad Alternatives: Contract out your website to a front-end web         developer
              Objections:I can't code
              Your startup’s better solution: Code-free website design tool         usable by anyone.
              Action Statement: Launch your website yourself.
              Header: Launch your website yourself. No coding required.

              Name of the Startup: Robinhood
              Bad Alternatives: High-fees on low volume trades.
              Objections:There's a minimum trade size
              Your startup’s better solution: No-fee stock trading platform
              Action Statement: Stock trading without fees.
              Header: Stock trading without fees. No trade minimums.

              Name of the Startup: Slack
              Bad Alternatives: Messy email chains and unsecure group chats.
              Objections:It'll cost too much
              Your startup’s better solution: Single app for real-time, team-        wide communication.
              Action Statement: Communicate with everyone in one place.
              Header: Communicate with everyone in one place. Free for teams.

              Name of the Startup: Bubble
              Bad Alternatives: Time consuming and expensive manual                   development by web development agencies
              Objections: I don't know how to code.
              Your startup’s better solution: Build the website using a simple       drag-drop UI without learning any code.
              Your Better Solution: Build your own website. Without code.
              Header: Build a custom website in 20 minutes. No code.

              Follow the below template for output. Do not exceed 10 words for each output. And introduce line breaks so that the 
              response appears one after the other.

              Hero Message 1: 
              Hero Message 2:
              Hero Message 3:
              """

    prompt = PromptTemplate(
        input_variables=["name", "bad_alternative", "better_solution", "objections", "icp"],
        template=template,
    )
    final_prompt = prompt.format(
        name=name,
        bad_alternative=bad_alternative,
        better_solution=better_solution,
        objections=objections,
        icp=icp
    )
    output_message = llm(final_prompt)


    #result = f"Result: {name}, {bad_alternative}, {better_solution}, {objections}, {icp}"
    return output_message

# Create your Streamlit app
def main():

    # Add a title
    st.title("HeroGPT")
    st.write("Create Stunning Hero Header Text")



    # Add some text inputs
    name = st.text_input("Enter the name of your startup (eg: Let us consider Airbnb as the example)")
    bad_alternative = st.text_input("What bad alternative do people resort to when they lack your product? (eg: Without Airbnb, people are stuck in sterile hotels, don't experience the real culture)")
    better_solution = st.text_input("How is your product better than that bad alternative? (eg: With Airbnb, stay in locals' homes and experience new cities like a local)")
    objections = st.text_input("State a top objection that the user might give as a reason and drop out without trying your product? (eg: In case of Airbnb, users may say that only long-term rentals are available on Airbnb and drop out or in case of Bubble.io, users may say that no-code app development may take too much time and a steep learning curve and end up dropping out.)")
    icp = st.text_input("Who is your ideal customer profile? (eg: Holiday and leisure travellers who stay in a Marriott, Hilton)")

    creativity_label = st.radio("Creativity", ["Low", "High"])
    creativity = 0.0 if creativity_label == "Low" else 0.7

    # Add a button to run the script and get output
    if st.button("Run script"):
        # Call your script with the user input
        result = run_script(name, bad_alternative, better_solution, objections, icp, creativity)

        # Display the output
        st.write(result)

# Run the app

st.markdown(
    """
    <a style='display: block; text-align: center; border: solid; color: white; background-color: #f63366; padding: 10px; margin: 10px;' href="https://www.producthunt.com/" target="_blank">Go to Product Hunt</a>
    """,
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    main()