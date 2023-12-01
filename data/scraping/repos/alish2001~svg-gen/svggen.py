# Step 1: Get simple rag working
import cohere
import os

import json
import uuid
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
import streamlit as st
from chatbot import Chatbot
from document import Documents
from directory import load_from_directory


api_key = "fkAeCp5ZzmMiI4YBtkKUD6BanZVdk1vImBGZ5W0m"
co = cohere.Client(api_key)


class App:
    def __init__(self, chatbot: Chatbot):
        """
        Initializes an instance of the App class.

        Parameters:
        chatbot (Chatbot): An instance of the Chatbot class.

        """
        self.chatbot = chatbot

    def run(self):
        """
        Runs the chatbot application.

        """
        # while True:
        # Get the user message using Streamlit input
        st.title("SVG-gen")
        message = st.text_input("User:", "")

        if st.button("Submit"):
            st.write("Sending request...")
            # Get the chatbot response
            response = self.chatbot.generate_response(message)

            # Print the chatbot response
            print("Chatbot:")
            flag = False
            svg_content = ""
            for event in response:
                # Text
                if event.event_type == "text-generation":
                    svg_content += event.text
                    print(event.text, end="")

                # Citations
                if event.event_type == "citation-generation":
                    if not flag:
                        print("\n\nCITATIONS:")
                        flag = True
                    print(event.citations)

            print(f"\n{'-'*100}\n")

            # discard everything but the <svg> tag
            # svg_content = svg_content.split("<svg")[1]
            # svg_content = "<svg" + svg_content

            svg_start = svg_content.find("<svg")
            svg_end = svg_content.find("</svg>") + len("</svg>")

            # Extracting the SVG content
            if svg_start != -1 and svg_end != -1:
                svg_content = svg_content[svg_start:svg_end]

            # Replace 'output.svg' with the actual path to your SVG file
            # svg_content = response
            print("DISPLAYING: ", svg_content)

            # Add custom CSS styles to constrain and center the SVG within the screen
            style = """
            <style>
                .svg-container {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    max-width: 50%;
                    max-height: 50vh;
                    margin: auto;
                }
                .svg-content {
                    width: 50%;
                    height: auto;
                }
            </style>
            """

            # Display the SVG content using st.markdown with custom styles
            st.markdown(style, unsafe_allow_html=True)
            st.markdown(
                '<div class="svg-container"><div class="svg-content">{}</div></div>'.format(
                    svg_content
                ),
                unsafe_allow_html=True,
            )


# load svg files from directory to embed
sources = load_from_directory("/Users/alish/Downloads/svgs")
# sources = [
#     {"title": "1 SVG", "file_path": "./sample-svgs/1.svg"},
#     {"title": "2 SVG", "file_path": "./sample-svgs/2.svg"},
#     {"title": "3 SVG", "file_path": "./sample-svgs/3.svg"},
#     {"title": "angle-up SVG", "file_path": "./sample-svgs/angle-up.svg"},
#     {"title": "bed-empty SVG", "file_path": "./sample-svgs/bed-empty.svg"},
# ]


# Create an instance of the Documents class with the given sources
documents = Documents(sources)

# Create an instance of the Chatbot class with the Documents instance
chatbot = Chatbot(documents)

# Create an instance of the App class with the Chatbot instance
app = App(chatbot)

# Run the chatbot
app.run()
