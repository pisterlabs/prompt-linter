"""
Common variables, paths etc.
"""

from credentials import API_KEY
import pickle
import openai

openai.api_key = API_KEY

SPACE_DATSET_ROOT = "/home/adithya/sem6/review-sum/datasets/space" 
SPACE_SAVE_DATA_ROOT = "/home/adithya/sem6/review-sum/saved-data/space"
PROMPT_DIR = "/home/adithya/sem6/review-sum/prompts"
GLOVE_PATH = "/home/adithya/sem6/review-sum/saved-data/glove_6B_100d.pkl"

GLOVE = pickle.load(open(GLOVE_PATH, 'rb'))['embeddings_index']

aspects_ = ["general", "rooms", "location", "service", "cleanliness", \
    "building", "food"]


if __name__ == '__main__':
    pass