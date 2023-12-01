import spacy
import openai
import textacy
from typing import List, Optional, Union, Callable
from spacy.tokens import Doc
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import Counter 
from wordcloud import WordCloud 
import matplotlib.pyplot as plt
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader 
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import create_extraction_chain
from langchain.chains import LLMChain


# load a pipeline package by name and return nlp object
nlp = spacy.load("en_core_web_trf", disable=["tok2vec","parser"])

def extract_entities(doc: Doc, 
                     include_types: Optional[Union[str, List[str]]] = None, 
                     sep: str = ' ') -> List[str]:
    """
    Extract named entities from a document and return them as strings.

    :param doc: The document to extract entities from.
    :param include_types: The types of entities to include. If None, include all types.
    :param sep: The separator to use when joining lemmas of multi-token entities.
    :return: A list of named entities in the form 'lemma/label'.
    """
    ents = textacy.extract.entities(doc, 
                                    include_types=include_types, 
                                    exclude_types=None, 
                                    drop_determiners=True, 
                                    min_freq=1)
    
    return [sep.join([token.text for token in entity])+'/'+entity.label_ for entity in ents]




def extract_named_entities_in_batches(df, entity_types=None, progress=None):
    """
    Processes a DataFrame in batches to extract named entities from text.

    Args:
    df (pd.DataFrame): Input DataFrame with text data.
    entity_types (list, optional): Specific entity types to extract.
    progress (streamlit.Progress, optional): Streamlit progress bar.

    Returns:
    pd.DataFrame: DataFrame with an additional 'named_entities' column.
    """
    # Define batch_size inside the function
    batch_size = 50

    # Calculate the number of batches
    batches = np.ceil(len(df) / batch_size).astype(int)

    named_entities = []

    # Loop over batches, step size is equal to batch size
    for i in tqdm(range(0, len(df), batch_size), total=batches):
        docs = nlp.pipe(df['text'][i:i+batch_size])
        
        for doc in docs:
            named_entities.append(extract_entities(doc, include_types=entity_types))

        # Update the progress bar, ensuring the value never exceeds 1.0
        if progress is not None:
            progress.progress(min((i + batch_size) / len(df), 1.0))

    df['named_entities'] = named_entities

    return df



def count_words(dataframe: pd.DataFrame, 
                column: str, 
                preprocess: Optional[Callable[[str], str]] = None, 
                min_frequency: int = 1) -> pd.DataFrame:
    """
    Count words in a specific column of a DataFrame.

    :param dataframe: The DataFrame to count words from.
    :param column: The column to count words in. Should be tokenized.
    :param preprocess: An optional function to preprocess the words before counting.
    :param min_frequency: The minimum frequency for a word to be included in the output.
    :return: A DataFrame sorted by word frequency, containing words and their frequencies.
    """
    word_counter = Counter()

    # If a preprocessing function is provided, apply it before counting words
    if preprocess:
        dataframe[column].map(lambda doc: word_counter.update(preprocess(doc)))
    else:
        dataframe[column].map(word_counter.update)

    # Convert Counter to DataFrame
    word_freq_df = pd.DataFrame.from_dict(word_counter, orient='index', columns=['freq'])
    
    # Filter words by minimum frequency
    word_freq_df = word_freq_df.query('freq >= @min_frequency')
    
    # Set index name for the dataframe
    word_freq_df.index.name = column

    # Sort DataFrame by frequency
    return word_freq_df.sort_values('freq', ascending=False)


def generate_word_cloud(data: pd.DataFrame,
                        col_name: str, 
                        max_words: int = 200):
    """
    Generate a word cloud from word frequencies.

    :param data: A pandas DataFrame containing text data.
    :param col_name: The column name to count words from.
    :param max_words: The maximum number of words in the word cloud.
    :return: A matplotlib Figure object containing the word cloud or a blank plot if no words were found.
    """
    word_frequencies = count_words(data, col_name).freq

    # Create new figure
    fig, ax = plt.subplots()

    # If word_frequencies is empty, return a blank plot
    if word_frequencies.empty:
        ax.text(0.5, 0.5, f'No words found to generate a word cloud for {col_name}.', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes)
        ax.axis('off')
        return fig

    # Convert pandas Series to Counter object
    word_frequencies = Counter(word_frequencies.fillna(0).to_dict())

    # Create wordcloud object
    word_cloud = WordCloud(width=800, height=400, 
                           background_color= "black", colormap="Paired", 
                           max_font_size=150, max_words=max_words)

    # Generate word cloud image from frequencies
    word_cloud.generate_from_frequencies(word_frequencies)

    # Display the cloud using matplotlib 
    ax.imshow(word_cloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(col_name.capitalize())

    # Return the figure
    return fig


def load_qa(file, openai_key, chain_type="stuff", k=5):
    """
    Create a QuestionAnswering chain with memory

    :param file: Text file
    :param chain_type: "stuff", "map_reduce", "refine", "map-rerank"
    :param max_words: The maximum number of words in the word cloud.
    :return: A matplotlib Figure object containing the word cloud or a blank plot if no words were found.
    """
    openai.api_key = openai_key    
    # load documents
    loader = CSVLoader(file_path=file,  encoding='utf-8')
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    # create vector database from data to use as index
    db = DocArrayInMemorySearch.from_documents(docs,
                                                embeddings)
    
    #  Keep a buffer of all prior messages 
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed internally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature = 0.0, openai_api_key=openai_key ,model_name='gpt-3.5-turbo-0613'), 
        chain_type=chain_type, 
        retriever=retriever, 
        memory=memory
    )
    return qa

def run_summarizer(file, openai_key, chain_type="map_reduce", model_name="gpt-3.5-turbo-0613"):
    """
    Create a summarizer chain
    """
    openai.api_key = openai_key
    # load documents
    with open(file) as f:
        full_text = f.read()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    splitted_texts = text_splitter.create_documents([full_text])
    # define embedding
    map_prompt = """
    Write a concise summary of the following text delimited by triple backquotes:
    
    ```{text}```
    
    - Capture the main points, themes, and key takeaways of the text
    - Ensure to include the most significant arguments, insights, and conclusions drawn from the text.
    - Ensure to include the timestamp when the spakers started talking about the main point.
    - Only respond with the timestamp and the concise summary, nothing else. 

    CONCISE SUMMARY:
    """
    map_prompt_template = PromptTemplate.from_template(map_prompt) # infer input variables automatically


    combine_prompt = """
    Write a concise summary of the following text delimited by triple backquotes.

    Return your response in bullet points which covers the key points of the text and \
    the timestamp when the spakers started talking about the main point. 
    ```{text}```
    BULLET POINT SUMMARY:
    """
    combine_prompt_template = PromptTemplate.from_template(combine_prompt) # infer input variables automatically
    
    summary_chain = load_summarize_chain(llm=ChatOpenAI(temperature = 0.0, openai_api_key=openai_key,model_name=model_name),
                                     chain_type=chain_type,
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template,
                                      verbose=False
                                    )
    episode_summary = summary_chain.run(splitted_texts)
    return episode_summary


def run_extraction_chain(episode_summary, openai_key, model_name="gpt-3.5-turbo-0613"):
    """
    Run extraction chain that extracts structured data from the summary
    """
    openai.api_key = openai_key

    # schema defines the properties you want to find and the expected types and description for those properties. 
    schema = {
        "properties": {
            # Summary of the text
            "summary": {
                "type": "string",
                "description" : "The concise summary of the text"
            },
            # Timestamp
            "timestamp": {
                "type": "string",
                "description" : "Timestamp when the spakers started talking about the topic"
            },
        },
        "required": ["summary", "timestamp"],
    }
    # Using gpt3.5 here because this is an easy extraction task and no need to jump to gpt4
    extraction_chain = create_extraction_chain(schema, 
                                               llm=ChatOpenAI(temperature = 0.0, openai_api_key=openai_key,model_name=model_name))
    summary_structured = extraction_chain.run(episode_summary)

    return summary_structured

def extract_topics(summary_structured, openai_key ,model_name="gpt-3.5-turbo-0613"):
    """
    Extract topics from the structured summary
    """
    openai.api_key = openai_key
    system_template = """
    You are a helpful assistant that helps retrieve topics talked about in a short text
    - You will be given a text
    - Your goal is to find the topic talked about in the text 
    - Only respond with the topic in 2 or 3 words, nothing else
    """
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template="Text: {text}" # Simply just pass the text as a human message
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    messages = [
        system_prompt,
        human_message_prompt,
    ]
    topic_prompt = ChatPromptTemplate.from_messages(messages)
    topic_chain = LLMChain(llm=ChatOpenAI(temperature = 0.0, openai_api_key=openai_key, model_name=model_name),
                           prompt=topic_prompt, 
                           verbose=False)
    
    # Holder for our topic timestamps
    timestamp_topic_dict = {}

    for element in summary_structured:
        

        text = f"{element['summary']}"
        topic = topic_chain.run(text)
        
        timestamp_topic_dict[element['timestamp']] = topic
        
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame.from_dict(list(timestamp_topic_dict.items()))

        # Rename the columns
        df.columns = ['Timestamp', 'Topic']

        # Remove duplicate 'Topic' entries
        df = df.drop_duplicates(subset="Topic", keep="first")
    return df