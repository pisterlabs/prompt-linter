import logging
import sys

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from core.util import movie_review_download

load_dotenv()
root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

placeholders = {
    "url0": "https://www.gulte.com/moviereviews/257842/miss-shetty-mr-polishetty-movie-review",
    "url1": "https://www.greatandhra.com/movies/reviews/miss-shetty-mr-polishetty-review-polishettys-show-131947",
    "url2": "https://www.telugubulletin.com/miss-shetty-mr-polishetty-movie-review-187727",
    "url3": "https://www.123telugu.com/reviews/miss-shetty-mr-polishetty-telugu-movie-review.html",
}

st.set_page_config(page_title="Movie Review")
st.sidebar.header("Movie Review Summarizer")

num_rows = st.slider("Number of rows", min_value=1, max_value=4)
my_form = st.form(key="form")
grid = my_form.columns(2)


# Function to create a row of widgets (with row number input to assure unique keys)
def add_row(row):
    with grid[0]:
        my_form.text_input(
            "URL", key=f"url{row}", placeholder=placeholders[f"url{row}"]
        )


# Loop to create rows of input widgets
for r in range(num_rows):
    add_row(r)

# Defining submit button
submit_button = my_form.form_submit_button(label="Submit")
if submit_button:
    st.write(st.session_state)
    prompt = [
        """
            Can you give me a Chris Stuckmann-style review from various sources in at most 200 words:
            """
    ]
    review_args = {}
    for r in range(num_rows):
        url = st.session_state[f"url{r}"]
        if url:
            print(url)
            review = movie_review_download(url)
            review_args[f"review{r + 1}"] = review
            prompt.append(f"{r + 1}. {{review{r + 1}}}")
            st.header(url)
            st.write(review)
    #
    prompt_template = ChatPromptTemplate.from_template("\n".join(prompt))
    chat = ChatOpenAI(temperature=0.0)
    prompt = prompt_template.format_messages(**review_args)
    customer_response = chat(prompt)
    if customer_response.content:
        st.header("Summarized Review")
        st.write(customer_response.content)
