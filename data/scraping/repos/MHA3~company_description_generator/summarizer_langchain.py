import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

llm_name = "gpt-4"
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.3, model=llm_name, openai_api_key=openai_api_key)


def preprocess(input_file_path: str) -> pd.DataFrame:
    """
    Reads data from a CSV file, cleans and translates the text in the DataFrame.

    Parameters:
    input_file_path (str): The path to the input CSV file.

    Returns:
    pd.DataFrame: A DataFrame with the cleaned and translated text.
    """
    # Import data
    df = pd.read_csv(input_file_path)

    # Set up cleaning chain
    cleaning_prompt = ChatPromptTemplate.from_template(
        "Clean the following text ```{text}```"
    )
    cleaning_chain = LLMChain(llm=llm, prompt=cleaning_prompt)

    # Set up translating chain
    translating_prompt = ChatPromptTemplate.from_template(
        "Translate the following text into english ```{text}```"
    )
    translating_chain = LLMChain(llm=llm, prompt=translating_prompt)

    # Process text
    preprocess_chain = SimpleSequentialChain(
        chains=[cleaning_chain, translating_chain],
        verbose=True
    )
    cleaned_descriptions = [preprocess_chain.run(text) for text in df["text"]]

    # Store results
    df["cleaned_text"] = cleaned_descriptions
    df.to_csv("./data/cleaned_descriptions.csv", index=False)
    return df


def generate_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates summaries from the cleaned and translated text in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the cleaned and translated text.

    Returns:
    pd.DataFrame: A DataFrame containing the generated summaries.
    """
    # Set up summarizing chain
    summarizing_prompt = ChatPromptTemplate.from_template(
        "Extract summary in the following format enclosed by single quotes"
        "'PROBLEM: describe the problem the company is trying to solve "
        "SOLUTION: company's proposed solution "
        "TARGET USERS: target users of the company "
        "OTHER DETAILS: other important details of the company', "
        "for the following company description enclosed by triple backticks "
        "```{company_description}```"
    )
    summarizing_chain = LLMChain(llm=llm, prompt=summarizing_prompt)

    # Generate summaries
    summaries = [summarizing_chain.run(description) for description in df["cleaned_text"]]

    # Store results
    df["summaries"] = summaries
    return df[["summaries"]]


if __name__ == "__main__":
    input_file_path = './data/input_sample.csv'
    cleaned_df = preprocess(input_file_path)
    summaries_df = generate_summaries(cleaned_df)
    summaries_df.to_csv("./data/out_langchain.csv", index=False)
