from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
import tempfile
import re

def main():



    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="Metro TransitTech ChatBot")
    st.header("TransitTech's ChatBot")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)

            # Save DataFrame to a temporary CSV file
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
                csv_path = tmp_file.name
                df.to_csv(csv_path, index=False)

            agent = create_csv_agent(
                OpenAI(temperature=0),
                csv_path,
                verbose=True
            )

            user_question = st.text_input("Ask a question about your CSV: ")

            if user_question is not None and user_question != "":
                with st.spinner(text="In progress..."):
                    st.write(agent.run(user_question))

        except Exception as e:
            st.error("API Key error.")
            st.error(str(e))

        if "can you show me where it is" in user_question:
            st.text("Yes, of couse! Give me 5 seconds at most")
            def extract_coordinates(input_text):
                try:
                    # Use regular expression to find latitude and longitude in the input text
                    # First, try to find the format "34.149917 latitude and -118.616783 longitude."
                    pattern1 = r'(-?\d+\.\d+)\s*latitude\s*and\s*(-?\d+\.\d+)\s*longitude\.'
                    match1 = re.search(pattern1, input_text)

                    # If the first format is not found, try to find the format "34.10155, -118.336883."
                    if not match1:
                        pattern2 = r'(-?\d+\.\d+),\s*(-?\d+\.\d+)'
                        match2 = re.search(pattern2, input_text)
                        if not match2:
                            raise ValueError
                        lat, lon = map(float, match2.groups())
                    else:
                        lat, lon = map(float, match1.groups())

                    return lat, lon

                except Exception as e:
                    st.error("Error extracting coordinates. Please enter a valid input.")
                    return None, None

            # Get user input
            user_input = agent.run(user_question)

            # Extract coordinates from the user input
            latitude, longitude = extract_coordinates(user_input)

            # Display the map with the extracted coordinates
            if latitude is not None and longitude is not None:
                df = pd.DataFrame({
                    'lat': [latitude],
                    'lon': [longitude]
                })

                st.map(df)





if __name__ == "__main__":
    main()