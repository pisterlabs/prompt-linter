import spacy
import stanza

from langchain.prompts.example_selector.coverage.substructs import get_dep_tree_spacy, get_dep_tree_stanza, tree_to_tuple

from datasets import load_dataset
from constants import Dataset as D
from driver import get_dataset, get_templates

dataset = D.BREAK
data_root = '../data'
ds = get_dataset(dataset, data_root)
templates = get_templates(dataset, 'Q-A')['example_templates']

for split in ds:
    trees = [tree_to_tuple(get_dep_tree_spacy(ex['question_text'])) for ex in ds[split]]