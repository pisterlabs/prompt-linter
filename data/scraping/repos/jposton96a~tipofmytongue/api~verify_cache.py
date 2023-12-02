import numpy as np
import openai

###########################
### FUNCTIONS
###########################


def load_embeddings(file_path):
    """
    Load the embeddings from the numpy array file.

    Args:
    - file_path (str): The path to the numpy array file.

    Returns:
    - embeddings (numpy.ndarray): The loaded embeddings.
    """
    print("Loading embeddings")
    data = np.load(file_path)
    embeddings = data["embeddings"]
    return embeddings


def load_word_dicts(file_path):
    """
    Load the words from the file.

    Args:
    - file_path (str): The path to the file containing the words.

    Returns:
    - lines (list): The lines read from the file.
    """
    print("Loading word dictionary")
    file = open(file_path, "r")
    lines = file.readlines()
    file.close()
    del file
    return lines


def count_populated(a: list[np.ndarray], prefix: bool = True):
    """
    Count the populated entries in a set of embeddings

    Args:
      - a: the input array
      - prefix: a boolean flag indicating whether to assume all populated elements are at the front

    Returns:
        _type_: _description_
    """
    count = 0
    for i, line in enumerate(a):
        if line.nonzero()[0].size == 0 or np.any(np.isnan(line)):
            # Count every time we encounter an empty cell
            count = count + 1

            # `prefix`=True:
            # Assumes all the populated elements are at the front, and
            # anything after an empty index will also be empty
            if prefix:
                return i

    # Return the final count
    return count

    # return len(a[np.count_nonzero(np.logical_not(np.logical_or(a == 0, np.isnan(a))), axis=1) > 0])


def test_spot_check_equality(embeddings, dictionary, indexes):
    for i in indexes:
        cache_embedding = embeddings[i]
        word = dictionary[i]

        response = openai.Embedding.create(model="text-embedding-ada-002", input=word)
        new_embedding = response["data"][0]["embedding"]

        print(cache_embedding[:10])
        print(new_embedding[:10])

        matches = np.array_equal(cache_embedding, new_embedding)
        matches_close = np.allclose(cache_embedding, new_embedding)
        print(
            f"Index {i} {'matches' if matches else 'does not match'} = (close={matches_close})"
        )

def test_n_offset(embeddings, dictionary, word_index, search_range=50):
    """
    Search neighboring elements for exact matches of the target word embedding to identify an offset error.

    Args:
        word_index (int): The index of the word in the dictionary.
        search_range (int, optional): The range of neighbors to check. Defaults to 50.

    Returns:
        None
    """
    word = dictionary[word_index]

    response = openai.Embedding.create(model="text-embedding-ada-002", input=word)
    new_embedding = response["data"][0]["embedding"]

    print(f"Checking neighbors for equality. Scanning for offset error += {search_range}. dict[{word_index}] = \"{word}\"")
    for i in range(word_index - search_range, word_index + search_range + 1):
        if i != word_index:
            cache_embedding = embeddings[i]
            matches = np.array_equal(cache_embedding, new_embedding)
            matches_close = np.allclose(cache_embedding, new_embedding)
            print(f"Index {i} {'matches' if matches else 'does not match'} = (close={matches_close})")

###########################
### Script
###########################

cache_path = "res/word_embeddings_cache.npz.chk_non_norm_311503"
dict_path = "res/words.txt"

embeddings = load_embeddings(cache_path)
dictionary = load_word_dicts(dict_path)
print(f"Loaded {len(dictionary)} words from {dict_path}")

# The length of the embeddings will always match the dictionary.
# Some or all of the indexes may be populated
embeddings_count = count_populated(embeddings)
print(f"Loaded {embeddings_count} embeddings from {cache_path}")
if embeddings_count == 0:
    print("Cache empty! Exiting")
    exit(-1)


#### Tests:

# Spot check the embeddings
indexes = list(range(0, embeddings_count, int(embeddings_count/10)))
test_spot_check_equality(embeddings, dictionary, indexes)

# I suspect an off by one error.
# Embed a word from the dictionary & scan near the word's index +-50 for a match.
word_index=1281
test_n_offset(embeddings, dictionary, word_index, search_range=50)