
import streamlit as st
from components import common
from utils import generate
import pandas as pd


def page():
    """
    Main function for the Streamlit app. This function handles user input,
    retrieves embeddings, and generates responses to user queries.

    When "Use Embeddings" is selected, the app retrieves embeddings for
    the query, finds the top-k closest metadata entries, creates an augmented
    query, and uses OpenAI's ChatCompletion API to generate a response. The
    URLs for the top-k metadata are also displayed as sources.

    When "Use Embeddings" is not selected, the app uses OpenAI's Completion
    API to generate a generic response to the query.
    """
    st.set_page_config(
        page_title="Embeddings",
        page_icon="👋",
    )
    common.sidebar()
    # create a sidebar
    st.sidebar.title("👈 Navigation")
    # Add links to different sections
    st.sidebar.markdown("[Word Embeddings Example](#word-embeddings-example)")
    st.sidebar.markdown(
        "[Multiple Word Embeddings Example](#multiple-words-example)")

    # create an input for embeddings
    # heading for embeddings
    st.write("# 📝 Embeddings")
    st.write("An embedding is a vector (list) of floating point numbers. The distance between two vectors measures their relatedness. Small distances suggest high relatedness and large distances suggest low relatedness.")
    st.write("To get started with embeddings, we need to understand how to measure the distance between two vectors. There are many ways to measure distance. The most common are Cosine Similarity,  Euclidean distance and Manhattan distance. The choice of distance metric depends on the specific use case and the nature of the embeddings. ")
    st.write("Example: The embedding for 'cat' is closer to the embedding for 'dog' than it is to the embedding for 'car'.")
    st.write("**Note**: To get started pls enter your API token in the sidebar 👈")
    # use get pass to hide the API key
    # get the user input
    openai_api_key = st.text_input(
        "OpenAI API Key", type="password")
    # save the API key in the session state
    st.session_state.openai_api_key = openai_api_key
    import os
    os.environ["OPENAI_API_KEY"] = openai_api_key

    st.write("## Word Embeddings Example")
    # write out a task list for the user
    # if the user selects the task, show the task
    st.write(
        "We can use OpenAI's API to generate embeddings for words. Let's try it out!")
    # Task 1: Generate embeddings for a word
    st.write("### Step 1: Generate Embeddings for a word")
    input_text = st.text_input("Enter a word to get started", "pizza")
    # button to generate embeddings
    if st.button("Generate Embeddings"):
        embeddings = generate.get_embeddings(input_text)
        st.write("**Request** OpenAI Python SDK code to generate embeddings")
        # show code
        code = """
            res = openai.Embedding.create(
                input=[query],
                engine="text-embedding-ada-002"
            )
            """
        st.code(code, language='python')
        # show in a show hide to show the embeddings
        st.write("**Response** OpenAI Embeddings API response")
        with st.expander("Show Code"):
            st.code(embeddings)
        st.balloons()

        st.write("The embeddings are generated using the text-embedding-ada-002 engine. This engine is trained on a large corpus of text and is able to generate embeddings for words, sentences and paragraphs.")
        st.write("*Note*: The embeddings are cached locally to speed up the process. The cache is stored in the cache folder. You can delete the cache folder to clear the cache.")

        st.write(
            f"Now that we have the embeddings for **{input_text}** we can use them to find the similarity between two words.")

        st.write("### Step 2: Compare", input_text, "with another word")
    if input_text:
        input_text_2 = st.text_input("Enter another word", "pasta")
        if st.button("Compare Similarity of words"):
            # get embeddings for both words
            embedding_1 = generate.get_embeddings(input_text)
            embedding_2 = generate.get_embeddings(input_text_2)
            # calculate similarity score
            similarity_score = generate.cosine_similarity(
                embedding_1[0]['embedding'], embedding_2[0]['embedding'])
            st.write("## Result ")
            st.write(
                f"The similarity score between is: {input_text} and {input_text_2} is", similarity_score)

            with st.expander("Show Code"):
                st.code(f"""
                # get_embeddings function
                def get_embeddings(query):
                    res = openai.Embedding.create(
                        input=[query],
                        engine="text-embedding-ada-002"
                    )
                    return res.data[0]['embedding']

                # get embeddings for both words {input_text} and {input_text_2}
                embedding_1 = get_embeddings('{input_text}')
                embedding_2 = get_embeddings('{input_text_2}')

                # calculate similarity score
                dot_product = sum(val1 * val2 for val1,
                                  val2 in zip(embedding1, embedding2))
                norm_vector1 = math.sqrt(sum(val * val for val in embedding1))
                norm_vector2 = math.sqrt(sum(val * val for val in embedding2))
                similarity = dot_product / (norm_vector1 * norm_vector2
                print(similarity)
                """, language='python')
    with st.expander("Show Code to get embeddings"):
        # show code
        code = """
            import openai
            openai.api_key = "sk-<your key>"
            res = openai.Embedding.create(
                input=[query],
                engine="text-embedding-ada-002"
            )
            print(res.data[0]['embedding'])
            """
        st.code(code, language='python')

    m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(2, 49, 244);
    color: white;
}
div.stButton > button:first-child:hover {
    background-color: rgb(18, 28, 45);
    color: white;
}

div.stButton > button:first-child:active {
        color: white;
    }
div.stButton > button:first-child:focus:not(:active) {
        background-color: rgb(18, 28, 45);
        color: white;
}
</style>""", unsafe_allow_html=True)

    # User input for word list
    st.write("## Multiple Words Example")
    st.write("We can also generate embeddings for a list of words and compare the similarity of that word against the others in the list.")
    st.write("### Step 1: Enter a list of words")
    word_input = st.text_area(
        "Enter a list of words", "burritos\npizza\npasta\nsushi\nchicken\nbeef\npork\nfish\nrice\nbeans\nbike")
    st.write("### Step 2: Enter a word to compare with the list of words")
    new_word = st.text_input("Enter a word", "")

    word_list = []
    if st.button("Compare Similarity of Words"):
        word_list = [word.strip() for word in word_input.split("\n")]
        # for each word, get the embeddings
        word_embeddings = [generate.get_embeddings(word) for word in word_list]
        with st.expander("Show Raw Embeddings"):
            st.write(word_embeddings)

        # st.success("Words added successfully!")
        st.write("## Result")
        st.write(f"Comparing {new_word} with {word_list}")
        # Calculate similarity scores and distances
        results = []
        for word in word_list:
            embedding_new = generate.get_embeddings(new_word)[0]['embedding']
            embedding_word = generate.get_embeddings(word)[0]['embedding']

            similarity_score = generate.cosine_similarity(
                embedding_new, embedding_word)

            euclidean_dist = generate.euclidean_distance(
                embedding_new, embedding_word)

            manhattan_dist = generate.manhattan_distance(
                embedding_new, embedding_word)

            results.append(
                (word, similarity_score, euclidean_dist, manhattan_dist))

        # Sort by similarity scores in descending order
        results = sorted(results, key=lambda x: x[1], reverse=True)

        # Sort by similarity scores in descending order
        results = sorted(results, key=lambda x: x[1], reverse=True)

        # st.write(
        st.write(
            f"The word with the highest similarity score is **{results[0][0]}** with a score of {results[0][1]}")

        # Create a DataFrame for the results
        df_results = pd.DataFrame(
            results, columns=["Word", "Cosine Similarity", "Euclidean Distance", "Manhattan Distance"])

        # Display results in a table
        st.table(df_results)

    with st.expander("Show Code to compare two words"):
        code = """
    import openai
    import math

    def get_embeddings(query):
        res = openai.Embedding.create(
                input=[query],
                engine="text-embedding-ada-002"
        )
        return res.data[0]['embedding']
    
    def cosine_similarity(embedding1, embedding2):
        dot_product = sum(val1 * val2 for val1,
                        val2 in zip(embedding1, embedding2))
        norm_vector1 = math.sqrt(sum(val * val for val in embedding1))
        norm_vector2 = math.sqrt(sum(val * val for val in embedding2))
        similarity = dot_product / (norm_vector1 * norm_vector2)
        return similarity

    embedding_new = get_embeddings(new_word)
    embedding_word = get_embeddings(word)

    similarity_score = cosine_similarity(embedding_new, embedding_word)

    """
        st.code(code, language='python')


if __name__ == "__main__":
    page()
