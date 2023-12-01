import streamlit as st
import openai
import jsonlines


def semantic_web(openai, input_query):

    try:
        search_response = openai.Engine("davinci").search(
            search_model="davinci",
            query=input_query,
            max_rerank=5,
            file="file-yyPiWKEFK9tf5vhFNF7A4gki",
            return_metadata=True
        )

        st.json(search_response.data)
        st.text(search_response.data[0].text)

    except:
        st.info("Something went wrong!")


def main():
    st.error("Deprecated!")
    # openai_key = st.text_input("Please enter OpenAI key here")
    st.header("semantic search over a set of documents.")
    st.info("Deprecated - [More info](https://platform.openai.com/docs/guides/search/search-deprecated)")
    st.write("""
    The Search endpoint (/search) allows you to do a semantic search over a set of documents. 
    This means that you can provide a query, such as a natural language question or a statement, and the provided documents will be scored and ranked based on how semantically related they are to the input query.
    """)

    # if openai_key:
    #     openai.api_key = openai_key

    #     st.header("Semantic Web")
    #     input_query = st.text_input("Enter your query", "neural")

    #     with st.expander("Resource file"):
    #         with jsonlines.open('asserts/resources.jsonl') as f:
    #             for line in f.iter():
    #                 st.text(line)

    #     if input_query:
    #         semantic_web(openai, input_query)

if __name__ == '__main__':
    main()
