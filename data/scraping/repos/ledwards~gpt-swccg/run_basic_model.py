import os
from dotenv import load_dotenv
import openai

load_dotenv()

openai.organization = "org-D2FBgBhwLFkKAOsgtSp86b4i"
openai.api_key = os.getenv("OPENAI_API_KEY")

remote_files = openai.File.list()["data"]
training_files = filter(lambda f: "answers.jsonl" in f["filename"], remote_files)
latest_file = max(training_files, key=lambda x: x["created_at"])

questions = [
    "What destiny is 'Luke Skywalker, Jedi Knight'?",
    "What planet is Luke Skywalker from?",
    "Which Dark Jedi Master has destiny 6?",
    "Which Dark Jedi is power 4 and destiny 6?",
    "How many vehicles have a maintenace icon?",
    "Which starship has a maintenace icon?",
    "What class of Star Destroyer is Conquest?",
    "Is Grand Moff Tarkin a leader?",
    "Is Grand Moff Tarkin a smuggler?",
]

for question in questions:
    answer = openai.Answer.create(
        search_model="ada", 
        model="curie", 
        question=question, 
        file=latest_file["id"], 
        examples_context="Captain Jean-Luc Picard is a Light Side character card. Captain Jean-Luc Picard is a Federation human. Captain Jean-Luc Picard has a power of 5. Will Riker is a humam. Will Riker has a power of 6. Data is an android. Data has a power of 10.",
        examples=[
            ["What Power is Jean-Luc Picard?", "Captain Jean Luc Picard is Power 5"],
            ["Which side of the Force is Picard?", "Picard is a Light Side card."],
            ["What race is Captain Jean-Luc Picard?", "Captain Jean-Luc Picard is human."],
            ["Is Jean-Luc Picard a Federation human?", "Yes"],
            ["Is Jean-Luc Picard a Dominion Changeling?", "No"],
            ["Which human has the highest power?", "Captain Jean-Luc Picard"],
            ["Which character has power 5?", "Captain Jean-Luc Picard"],
            ["Which card has the highest power?", "Data"],
            ["Which Federation character has the highest power?", "Data"]
        ], 
        max_rerank=50,
        max_tokens=20,
        stop=["\n", "<|endoftext|>"]
    )
    print(question)
    print(f'> {answer["answers"][0]}')