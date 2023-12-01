import os, time
import pandas as pd
import numpy as np
from collections import Counter
from .utils import abbreviate_book_name_in_full_reference, get_train_test_split_from_verse_list, embed_batch
from .types import TranslationTriplet, ChatResponse, VerseMap, AIResponse
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Callable
from random import shuffle
import requests
import guidance
import lancedb
from lancedb.embeddings import with_embeddings
from nltk.util import ngrams
from nltk import FreqDist 

import logging
logger = logging.getLogger('uvicorn')

machine = 'http://192.168.1.76:8081'

def get_dataframes(target_language_code=None, file_suffix=None):
    """Get source data dataframes (literalistic english Bible and macula Greek/Hebrew)"""
    bsb_bible_df = pd.read_csv('data/bsb-utf8.txt', sep='\t', names=['vref', 'content'], header=0)
    bsb_bible_df['vref'] = bsb_bible_df['vref'].apply(abbreviate_book_name_in_full_reference)
    macula_df = pd.read_csv('data/combined_greek_hebrew_vref.csv') # Note: csv wrangled in notebook: `create-combined-macula-df.ipynb`
    
    if target_language_code:
        target_tsv = get_target_vref_df(target_language_code, file_suffix=file_suffix)
        target_df = get_target_vref_df(target_language_code, file_suffix=file_suffix)
        return bsb_bible_df, macula_df, target_df

    else:
        return bsb_bible_df, macula_df

def get_vref_list(book_abbreviation=None):
    vref_url = 'https://raw.githubusercontent.com/BibleNLP/ebible/main/metadata/vref.txt'
    if not os.path.exists('data/vref.txt'):
        os.system(f'wget {vref_url} -O data/vref.txt')

    with open('data/vref.txt', 'r', encoding="utf8") as f:
        
        if book_abbreviation:
            return [i.strip() for i in f.readlines() if i.startswith(book_abbreviation)]
        
        else:
            return list(set([i.strip().split(' ')[0] for i in f.readlines()]))

def get_target_vref_df(language_code, file_suffix=None, drop_empty_verses=False):
    """Get target language data by language code"""
    if not len(language_code) == 3:
        return 'Invalid language code. Please use 3-letter ISO 639-3 language code.'
    
    language_code = language_code.lower().strip()
    
    language_code = f'{language_code}-{language_code}'
    # if file_suffix:
    #     print('adding file suffix', file_suffix)
    language_code = f'{language_code}{file_suffix if file_suffix else ""}'
    
    target_data_url = f'https://raw.githubusercontent.com/BibleNLP/ebible/main/corpus/{language_code}.txt'
    path = f'data/{language_code}.txt'
    
    if not os.path.exists(path):
        try:
            os.system(f'wget {target_data_url} -O {path}')
        except:
            return 'No data found for language code. Please check the eBible repo for available data.'

    with open(path, 'r', encoding="utf8") as f:
        target_text = f.readlines()
        target_text = [i.strip() for i in target_text]

    vref_url = 'https://raw.githubusercontent.com/BibleNLP/ebible/main/metadata/vref.txt'
    if not os.path.exists('data/vref.txt'):
        os.system(f'wget {vref_url} -O data/vref.txt')

    with open('data/vref.txt', 'r', encoding="utf8") as f:
        target_vref = f.readlines()
        target_vref = [i.strip() for i in target_vref]

    target_tsv = [i for i in list(zip(target_vref, target_text))]
    
    if drop_empty_verses:
        target_tsv = [i for i in target_tsv if i[1] != '']
    
    target_df = pd.DataFrame(target_tsv, columns=['vref', 'content'])
    
    return target_df

from pandas import DataFrame as DataFrameClass

def create_lancedb_table_from_df(df: DataFrameClass, table_name, content_column_name='content'):
    """Turn a pandas dataframe into a LanceDB table."""
    start_time = time.time()
    logger.info('Creating LanceDB table...')
    import lancedb
    from lancedb.embeddings import with_embeddings
    
    logger.error(f'Creating LanceDB table: {table_name}, {df.head}')
    
    # rename 'content' field as 'text' as lancedb expects
    try:
        df = df.rename(columns={content_column_name: 'text'})
    except:
        assert 'text' in df.columns, 'Please rename the content column to "text" or specify the column name in the function call.'
    
    # Add target_language_code to the dataframe
    df['language_code'] = table_name
    
    # mkdir lancedb if it doesn't exist
    if not os.path.exists('./lancedb'):
        os.mkdir('./lancedb')
    
    # Connect to LanceDB
    db = lancedb.connect("./lancedb")
    
    table = get_table_from_database(table_name)
    
    if not table:
        # If it doesn't exist, create it
        df_filtered = df[df['text'].str.strip() != '']
        # data = with_embeddings(embed_batch, df_filtered.sample(1000)) # FIXME: I can't process the entirety of the bsb bible for some reason. Something is corrupt or malformed in the data perhaps
        data = with_embeddings(embed_batch, df_filtered) 

        # data = with_embeddings(embed_batch, df)
        
        table = db.create_table(
            table_name,
            data=data,
            mode="create",
        )
    else:
        # If it exists, append to it
        df_filtered = df[df['text'].str.strip() != '']
        data = with_embeddings(embed_batch, df_filtered.sample(10000))
        data = data.fillna(0)  # Fill missing values with 0
        table.append(data)
    
    print('LanceDB table created. Time elapsed: ', time.time() - start_time, 'seconds.')
    return table  

def load_database(target_language_code=None, file_suffix=None):
    print('Loading dataframes...')
    if target_language_code:
        print(f'Loading target language data for {target_language_code} (suffix: {file_suffix})...')
        bsb_bible_df, macula_df, target_df = get_dataframes(target_language_code, file_suffix=file_suffix)
    else:
        print('No target language code specified. Loading English and Greek/Hebrew data only.')
        bsb_bible_df, macula_df = get_dataframes()
        target_df = None
    
    print('Creating tables...')
    # table_name = 'verses'
    # create_lancedb_table_from_df(bsb_bible_df, table_name)
    # create_lancedb_table_from_df(macula_df, table_name)
    create_lancedb_table_from_df(bsb_bible_df, 'bsb_bible')
    create_lancedb_table_from_df(macula_df, 'macula')
    
    if target_df is not None:
        print('Creating target language tables...')
        # create_lancedb_table_from_df(target_df, table_name)
        target_table_name = target_language_code if not file_suffix else f'{target_language_code}{file_suffix}'
        create_lancedb_table_from_df(target_df, target_table_name)

    print('Database populated.')
    return True
    
def get_table_from_database(table_name):
    """
    Returns a table by name. 
    Use '/api/db_info' endpoint to see available tables.
    """
    import lancedb
    db = lancedb.connect("./lancedb")
    table_names = db.table_names()
    if table_name not in table_names:
        logger.error(f'''Table {table_name} not found. Please check the table name and try again.
                     Available tables: {table_names}''')
        return None

    table = db.open_table(table_name)
    return table

def get_verse_triplet(full_verse_ref: str, language_code: str, bsb_bible_df, macula_df):
    """
    Get verse from bsb_bible_df, 
    AND macula_df (greek and hebrew)
    AND target_vref_data (target language)
    
    e.g., http://localhost:3000/api/verse/GEN%202:19&aai
    or NT: http://localhost:3000/api/verse/ROM%202:19&aai
    """
    
    bsb_row = bsb_bible_df[bsb_bible_df['vref'] == full_verse_ref]
    macula_row = macula_df[macula_df['vref'] == full_verse_ref]
    target_df = get_target_vref_df(language_code)
    target_row = target_df[target_df['vref'] == full_verse_ref]
    
    if not bsb_row.empty and not macula_row.empty:
        return {
            'bsb': {
                'verse_number': int(bsb_row.index[0]),
                'vref': bsb_row['vref'][bsb_row.index[0]],
                'content': bsb_row['content'][bsb_row.index[0]]
            },
            'macula': {
                'verse_number': int(macula_row.index[0]),
                'vref': macula_row['vref'][macula_row.index[0]],
                'content': macula_row['content'][macula_row.index[0]]
            },
            'target': {
                'verse_number': int(target_row.index[0]),
                'vref': target_row['vref'][target_row.index[0]],
                'content': target_row['content'][target_row.index[0]]
            }
        }
    else:
        return None

def query_lancedb_table(language_code: str, query: str, limit: str='50'):
    """Get similar sentences from a LanceDB table."""
    # limit = int(limit) # I don't know if this is necessary. The FastAPI endpoint might infer an int from the query param if I typed it that way
    table = get_table_from_database(language_code)
    query_vector = embed_batch([query])[0]
    if not table:
        return {'error':'table not found'}
    result = table.search(query_vector).limit(limit).to_df().to_dict()
    if not result.values():
        return []
    texts = result['text']
    # scores = result['_distance']
    vrefs = result['vref']
    
    output = []
    for i in range(len(texts)):
        output.append({
            'text': texts[i],
            # 'score': scores[i],
            'vref': vrefs[i]
        })
        
    return output
    
def get_unique_tokens_for_language(language_code):
    """Get unique tokens for a language"""
    tokens_to_ignore = ['']
    
    if language_code == 'bsb' or language_code =='bsb_bible':
        df, _, _ = get_dataframes()
    elif language_code =='macula':
        _, df, _ = get_dataframes()
    else:
        _, _, df = get_dataframes(target_language_code=language_code)
        
    target_tokens = df['content'].apply(lambda x: x.split(' ')).explode().tolist()
    target_tokens = [token for token in target_tokens if token not in tokens_to_ignore]
    unique_tokens = Counter(target_tokens)
    return unique_tokens

def get_ngrams(language_code: str, size: int=2, n=100, string_filter: list[str]=[]):
    """Get ngrams with frequencies for a language
    
    Params: 
    - language_code (str): language code
    - size (int): ngram size
    - n (int): max number of ngrams to return
    - string_filter (list[str]): if passed, only return ngrams where all ngram tokens are contained in string_filter
    
    A string_filter might be, for example, a tokenized sentence where you want to detect ngrams relative to the entire corpus.
    
    NOTE: calculating these is not slow, and it is assumed that the corpus itself will change during iterative translation
    If it winds up being slow, we can cache the results and only recalculate when the corpus changes. # ?FIXME
    """
    tokens_to_ignore = ['']
    # TODO: use a real character filter. I'm sure NLTK has something built in
    
    if language_code == 'bsb' or language_code =='bsb_bible':
        df, _, _ = get_dataframes()
    elif language_code =='macula':
        _, df, _ = get_dataframes()
    else:
        _, _, df = get_dataframes(target_language_code=language_code)
    
    target_tokens = df['content'].apply(lambda x: x.split(' ')).explode().tolist() 
    target_tokens = [token for token in target_tokens if token not in tokens_to_ignore]
    
    n_grams = [tuple(gram) for gram in ngrams(target_tokens, size)]

    print('ngrams before string_filter:', len(n_grams))

    if string_filter:
        print('filtering with string_filter')
        n_grams = [gram for gram in n_grams if all(token in string_filter for token in gram)]

    freq_dist = FreqDist(n_grams)

    print('ngrams after string_filter:', len(n_grams))
    
    return list(freq_dist.most_common(n))

def build_translation_prompt(
        vref, 
        target_language_code, 
        source_language_code=None, 
        bsb_bible_df=None, 
        macula_df=None, 
        number_of_examples=3, 
        backtranslate=False) -> dict[str, TranslationTriplet]:
    
    """Build a prompt for translation"""
    if bsb_bible_df is None or bsb_bible_df.empty or macula_df is None or macula_df.empty: # build bsb_bible_df and macula_df only if not supplied (saves overhead)
        bsb_bible_df, macula_df, target_df = get_dataframes(target_language_code=target_language_code)
    if source_language_code:
        _, _, source_df = get_dataframes(target_language_code=source_language_code)
    else:
        source_df = bsb_bible_df
    
    # Query the LanceDB table for the most similar verses to the source text (or bsb if source_language_code is None)
    table_name = source_language_code if source_language_code else 'bsb_bible'
    query = source_df[source_df['vref']==vref]['content'].values[0]
    original_language_source = macula_df[macula_df['vref']==vref]['content'].values[0]
    print(f'Query result: {query}')
    similar_verses = query_lancedb_table(table_name, query, limit=number_of_examples) # FIXME: query 50 and then filter to first n that have target content?
    
    triplets = [get_verse_triplet(similar_verse['vref'], target_language_code, bsb_bible_df, macula_df) for similar_verse in similar_verses]
    
    target_verse = target_df[target_df['vref']==vref]['content'].values[0]
    
    # Initialize an empty dictionary to store the JSON objects
    json_objects: dict[str, TranslationTriplet] = dict()
    
    for triplet in triplets:
        # Create a JSON object for each triplet with top-level keys being the VREFs
        json_objects[triplet["bsb"]["vref"]] = TranslationTriplet(
            source=triplet["macula"]["content"],
            bridge_translation=triplet["bsb"]["content"],
            target=triplet["target"]["content"] # FIXME: validate that content exists here?
        ).to_dict()
    
    # Add the source verse Greek/Hebrew and English reference to the JSON objects
    json_objects[vref] = TranslationTriplet(
        source=original_language_source,
        bridge_translation=query,
        target=target_verse
    ).to_dict()
        
    return json_objects


def execute_discriminator_evaluation(verse_triplets: dict[str, TranslationTriplet], hypothesis_vref: str, hypothesis_key='target') -> ChatResponse:
    """
    Accepts an array of verses as verse_triplets.
    The final triplet is assumed to be the hypothesis.
    The hypothesis string is assumed to be the target language rendering.
    
    This simple discriminator type of evaluation scrambles the input verse_triplets
    and prompts the LLM to detect which is the hypothesis.
    
    The return value is:
    {
        'y_index': index_of_hypothesis,
        'y_hat_index': llm_predicted_index,
        'rationale': rationale_string,
    }
    
    If you introduce any intermediate translation steps (e.g., leaving unknown tokens untranslated),
    then this type of evaluation is not recommended.
    """
    hypothesis_triplet = verse_triplets[hypothesis_vref]
    print(f'Hypothesis: {hypothesis_triplet}')
    
    verse_triplets_list: list[tuple] = list(verse_triplets.items())
    
    print('Verse triplets keys:', [k for k, v in verse_triplets_list])
    # # Shuffle the verse_triplets
    shuffle(verse_triplets_list)
    print(f'Shuffled verse triplets keys: {[k for k, v in verse_triplets_list]}')
    
    # # Build the prompt
    prompt = ''
    for i, triplet in enumerate(verse_triplets_list):
        print(f'Verse triplet {i}: {triplet}')
        prompt += f'\n{triplet[0]}. Target: {triplet[1]["target"]}'

    url = f"{machine}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            # FIXME: I think I should just ask the model to designate which verse stands out as the least likely to be correct.
            {"role": "user", "content": f"### Instruction: One of these translations is incorrect, and you can only try to determine by comparing the examples given:\n{prompt}\nWhich one of these is incorrect? (show only '[put verse ref here] -- rationale as to why you picked this one relative only to the other options')\n###Response:"}
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False,
    }

    response = requests.post(url, json=payload, headers=headers)
    
    return response.json()

def execute_fewshot_translation(vref, target_language_code, source_language_code=None, bsb_bible_df=None, macula_df=None, number_of_examples=3, backtranslate=False) -> ChatResponse:
    prompt = build_translation_prompt(vref, target_language_code, source_language_code, bsb_bible_df, macula_df, number_of_examples, backtranslate)
    url = f"{machine}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": False,
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

class RevisionLoop(BaseModel):
    # FIXME: this loop should only work for (revise-evaluate)*n, where you start with a translation draft.
    # TODO: implement a revision function whose output could be evaluated
    iterations: int
    function_a: Optional[Callable] = None
    function_b: Optional[Callable] = None
    function_a_output: Optional[Any] = Field(None, description="Output of function A")
    function_b_output: Optional[Any] = Field(None, description="Output of function B")
    loop_data: Optional[List[Any]] = Field(None, description="List to store data generated in the loop")
    current_iteration: int = Field(0, description="Current iteration of the loop")

    def __init__(self, iterations: int, function_a=execute_fewshot_translation, function_b=execute_discriminator_evaluation):
        super().__init__(iterations=iterations)
        self.function_a = function_a
        self.function_b = function_b
        self.loop_data = ['test item']

    def __iter__(self):
        self.current_iteration = 0
        return self

    def __next__(self):
        if self.current_iteration < self.iterations:
            print("Executing function A...")
            self.function_a_output: VerseMap = self.function_a()
            print("Executing function B...")
            # inputs for function b: (verse_triplets: dict[str, TranslationTriplet], hypothesis_vref: str, hypothesis_key='target') -> ChatResponse:
            function_b_input = {
                "verse_triplets": self.function_a_output,
                "hypothesis_vref": list(self.function_a_output.keys())[-1],
                "hypothesis_key": "target"
            }
            self.function_b_output = self.function_b(**function_b_input)
            self.loop_data.append((self.function_a_output, self.function_b_output))
            self.current_iteration += 1
            return self.function_a_output, self.function_b_output
        else:
            print("Reached maximum iterations, stopping loop...")
            raise StopIteration


    def get_loop_data(self):
        return self.loop_data
    
class Translation():
    """Translations differ from revisions insofar as revisions require an existing draft of the target"""
    
    def __init__(self, vref: str, target_language_code: str, number_of_examples=3, should_backtranslate=False):
        self.vref = vref
        self.target_language_code = target_language_code
        self.number_of_examples = number_of_examples
        self.should_backtranslate = should_backtranslate
        
        bsb_bible_df, macula_df = get_dataframes()
        self.verse = get_verse_triplet(full_verse_ref=self.vref, language_code=self.target_language_code, bsb_bible_df=bsb_bible_df, macula_df=macula_df)
        self.vref_triplets = build_translation_prompt(vref, target_language_code)
        # Predict translation
        self.hypothesis: ChatResponse = execute_fewshot_translation(vref, target_language_code, source_language_code=None, bsb_bible_df=bsb_bible_df, macula_df=macula_df, number_of_examples=3, backtranslate=False)
        # Get feedback on the translation
        # NOTE: here is where various evaluation functions could be swapped out
        self.feedback: ChatResponse = execute_discriminator_evaluation(self.vref_triplets, self.vref)

    def get_hypothesis(self):
        return self.hypothesis
    
    def get_feedback(self):
        return self.feedback
