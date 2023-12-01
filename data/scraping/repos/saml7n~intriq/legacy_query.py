import json
import re
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string


file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


@st.cache_data(ttl="2h")
def load_data(uploaded_files):
    dataframes = {}
    for uploaded_file in uploaded_files:
        try:
            ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
        except:
            ext = uploaded_file.split(".")[-1]
        if ext in file_formats:
            if ext in ['xls', 'xlsx', 'xlsm', 'xlsb']:
                # For Excel files, read each sheet as a separate dataframe
                with pd.ExcelFile(uploaded_file) as xls:
                    for sheet_name in xls.sheet_names:
                        dataframes[f'{uploaded_file.name}-{sheet_name}'] = pd.read_excel(
                            xls, sheet_name)
            else:
                # For other file types
                dataframes[uploaded_file.name] = file_formats[ext](
                    uploaded_file)
        else:
            st.error(f"Unsupported file format: {ext}")
    return dataframes


#################################
# Download necessary NLTK data
nltk.download('stopwords')


def preprocess_text(text):
    """Basic text preprocessing to remove punctuation and stopwords."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return ' '.join([word for word in text.split() if word not in stopwords.words('english')])


def analyse_columns(df_info_dict, query):
    """
    Analyze a dictionary of column descriptions to find relevant columns based on the query.
    """
    relevant_columns = []

    # Preprocess query
    query = preprocess_text(query)

    # Prepare descriptions and their corresponding column names
    descriptions = [preprocess_text(desc) for desc in df_info_dict.keys()]
    column_names = list(df_info_dict.values())

    # Check for content matches using TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(descriptions)
    query_vec = tfidf.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix)

    # Adding relevant columns based on similarity threshold
    for idx, similarity in enumerate(similarities[0]):
        if similarity > 0.1:  # Threshold for relevance
            relevant_columns.append(column_names[idx])

    return relevant_columns
#########################################################


def create_df_summary(llm, df):
    prompt = '''
    Create me a dict of summary statistics for each column of this dataframe.
    This should be formatted as a python dict where the value is the column name and the key is a 5 word summary of the column based on the information in the column.
    '''
    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )
    response = pandas_df_agent.run(
        st.session_state.messages
    )
    return extract_dictionary_from_response(response)


def extract_dictionary_from_response(response):
    # Regular expression to find dictionary-like patterns
    dict_pattern = r'\{.*?\}'

    # Search for the pattern
    matches = re.findall(dict_pattern, response, re.DOTALL)

    # Assuming the first match is the desired dictionary
    if matches:
        try:
            # Convert the string to a dictionary
            dict_response = json.loads(matches[0])
            return dict_response
        except json.JSONDecodeError:
            # Handle cases where the extraction does not result in valid JSON
            print("Extracted text is not a valid JSON.")
            return None
    else:
        print("No dictionary pattern found in the response.")
        return None


def create_info_dict(llm, dataframes):
    df_info_dicts = []
    for df_name, df in dataframes.items():
        col_summary_dict = create_df_summary(llm, df)
        for key, val in col_summary_dict.items():
            col_summary_dict[val] = (df_name, val)
        df_info_dicts.append(col_summary_dict)
    return pd.concat(df_info_dicts)


# def determine_relevant_columns(llm, df_info, user_query):
#     prompt = '''
#     Create me a dict of summary statistics for each column of this dataframe.
#     This should be formatted as a python dict where the key is the column name and the value is a 5 word summary of the column based on the information in the column.
#     '''

#     pandas_df_agent = create_pandas_dataframe_agent(
#         llm,
#         df,
#         verbose=True,
#         agent_type=AgentType.OPENAI_FUNCTIONS,
#         handle_parsing_errors=True,
#     )
#     response = pandas_df_agent.run(
#         st.session_state.messages, callbacks=[st_cb]
#     )
#     if col:
#         return extract_dictionary_from_response(response)
#     else:
#         return response


st.set_page_config(
    page_title="LangChain: Chat with pandas DataFrame", page_icon="🦜")
st.title("🦜 LangChain: Chat with pandas DataFrame")

uploaded_file = st.file_uploader(
    "Upload up to 5 data files",
    type=list(file_formats.keys()),
    accept_multiple_files=True,
    help="Various File formats are Support",
    on_change=clear_submit,
)

if not uploaded_file:
    st.warning(
        "This app uses LangChain's `PythonAstREPLTool` which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app."
    )

if uploaded_file:
    dfs = load_data(uploaded_file)

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(
        temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key, streaming=True
    )

    df_info_dict = create_info_dict(llm, dfs)

    # # Process user query
    # relevant_columns = determine_relevant_columns(
    #     llm, df_info, df_col_info, prompt)

    concated_df = pd.concat(dfs.values(), axis=1)
    relevant_columns = analyse_columns(llm, concated_df, prompt)

    contextual_df = pd.DataFrame()
    for df_name, cols in relevant_columns.items():
        if df_name in dfs:
            contextual_df = pd.concat(
                [contextual_df, dfs[df_name][cols]], axis=1)

    pandas_df_agent = create_pandas_dataframe_agent(
        llm,
        contextual_df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
            st.container(), expand_new_thoughts=False)
        response = pandas_df_agent.run(
            st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
        st.write(response)
