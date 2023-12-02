import os
from dotenv import load_dotenv
import openai
dirname = os.path.dirname(__file__)
load_dotenv()

openai.organization = "org-D2FBgBhwLFkKAOsgtSp86b4i"
openai.api_key = os.getenv("OPENAI_API_KEY")

file = os.path.join(dirname, '../data/search.jsonl')
openai.File.create(file=open(file), purpose='search')

file = os.path.join(dirname, '../data/answers.jsonl')
openai.File.create(file=open(file), purpose='answers')

file = os.path.join(dirname, '../data/fine_tune.jsonl')
openai.File.create(file=open(file), purpose='fine-tune')

file = os.path.join(dirname, '../data/fine_tune_validation.jsonl')
openai.File.create(file=open(file), purpose='fine-tune')
