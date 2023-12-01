import openai
import openai_key #importing api key file
import config
openai.api_key=openai_key.key
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def clean_up_gpt_response_list(message):
    """
    this function acceps a gpt response as a list and parses it as a pythonic list. 
    if the response is an invalid response, it returns an empty list
    """
    messages=[
        {"role": "system", "content": config.clean_up_prompt_list},
        {"role": "user", "content": f"gpt generated text:\n```{message}```"},
    ]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages,temperature=0) #temp=0 since we do not want newer words
    try:
        list_of_words=json.loads(response["choices"][0]["message"]["content"])['list_of_words']
    except:
        print('error parsing list')
        list_of_words=[]
    return list_of_words

def clean_up_gpt_response_clue(message):
    """
    this function accepts a gpt response for clue generation and parses it to return a dictionary with the Clue. 
    These functions can be replaced with more tradidional functions that find the required data inside our response and
    also with better prompting.
    it returns an empty string if parsing the response fails.
    """
    messages=[
        {"role": "system", "content": config.clean_up_prompt_clue},
        {"role": "user", "content": f"gpt generated text:\n```{message}```"},
    ]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=messages,temperature=0)
    try:
        clue=json.loads(response["choices"][0]["message"]["content"])['clue']
    except:
        print('error parsing clue')
        clue=''
    return clue

def generate_words(theme,max_word_limit):
    """
    this function accepts a theme and a limit for the maximum length of words needed for a crossword and calls GPT3.5 to create a 
    list of 20 words (which are then filtered by length) that can be utilised in the crossword.
    the sampling penalties ensure that the GPT generated words are varied but also common to the theme.
    """
    messages=[{"role": "system", "content": config.list_generator_prompt},
              {"role": "user", "content": f"<theme>: {theme} \n<maximum word length>: {max_word_limit}"}]
    response = openai.ChatCompletion.create(model=config.list_generator_model,messages=messages,
                                            frequency_penalty=config.list_generator_frequency_penalty,
                                            presence_penalty=config.list_generator_presence_penalty,
                                            temperature=config.list_generator_temp)
    generated_words=clean_up_gpt_response_list(response["choices"][0]["message"]["content"])
    #refiltering as some words will inevitably be of higher length than our max word limit
    word_list=[word for word in generated_words if len(word)<=max_word_limit]
    return word_list

def generate_clue(theme,word):
    """
    this function is responsible for generating clues ensuring that a chain-of-thought is established for the solver.
    the model we have used here is GPT 3.5 but GPT4 or a more fine-tuned word-clue model might yield higher quality results.
    """
    messages=[{"role": "system", "content": config.clue_generator_prompt},
              {"role": "user", "content": f"<theme>: {theme} \n<word>: {word}"}]
    response = openai.ChatCompletion.create(model=config.clue_generator_model,messages=messages,temperature=config.clue_generator_temp)
    generated_clue=clean_up_gpt_response_clue(response["choices"][0]["message"]["content"])
    
    return generated_clue

def generate_all_clues(word_list,theme):
    """
    function responsible for generating clues for a list of words "word_list" according to a certain "theme".
    this function can be improved and made more efficient with async API calls.
    returns a dictionary with values as the clues and keys as the words. {'word1':'clue1','word2':'clue2'......}
    """
    clues={}
    for word in word_list:
        clue=generate_clue(theme,word)
        clues[word]=clue
    return clues

def sort_according_to_rank(word_list,theme_phrase):
    """
    this function accepts a list of words 'word_list' and 'theme_phrase' and generates BERT based sentence embeddings.
    the word list is then sorted according to the cosine similarity of the words with the provided theme.
    the returned list of words also has the rank of the words appended to the words itself, 
    eg: if 'horse' (an element in our list) is ranked 1st closest and 'stable' is ranked 3rd closest to our theme_phrase='animals', 
    then the function will output ['1 - horse',...,'3 - stable',...]
    """
    model = SentenceTransformer('model')
    # Generate embeddings for the theme phrase and word list
    theme_embedding = model.encode([theme_phrase], convert_to_tensor=True)  # Reshape to 2D array
    word_embeddings = model.encode(word_list, convert_to_tensor=True)
    # Calculate cosine similarity between theme embedding and word embeddings
    similarities = cosine_similarity(theme_embedding, word_embeddings)
    # Create a dictionary to store words and their corresponding similarity scores
    ranked_words = {word: similarity.item() for word, similarity in zip(word_list, similarities[0])}
    # Sort the words based on their similarity scores in descending order
    ranked_words = {k: v for k, v in sorted(ranked_words.items(), key=lambda item: item[1], reverse=True)}
    ordered_words=list(ranked_words.keys())
    indexed_words = [f"{str(index+1)} - "+word for index, word in enumerate(ordered_words)]
    return indexed_words

def remove_ranks(list_of_words):
    """
    this function removes the ranks appended to words in a list "list_of_words"
    eg. ['1 - horse',...,'3 - stable',...] -> ['horse',...,'stable',...]
    """
    return [word.split(' ')[-1] for word in list_of_words]