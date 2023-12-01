import openai
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY_1")

jokes = pd.read_csv(
    "../other/jokes/cleaned_compiled_data/final_cleaned_jokes.csv")

training_jokes = pd.DataFrame(columns=["prompt", "completion"])
testing_jokes = []

if not os.path.exists("../other/jokes/cleaned_compiled_data/fine_tune_dataset.json"):
    for index, joke in jokes.iterrows():
        if not pd.isna(joke["category"]):
            training_jokes = pd.concat([training_jokes, pd.DataFrame({
                "completion": [joke["joke"]],
                "prompt": [f'Joke']
            })])

        else:
            testing_jokes.append(joke)

    training_jokes.to_json(
        "../other/jokes/cleaned_compiled_data/fine_tune_dataset.json", orient="records")

print("Training jokes: ", len(training_jokes))
if not os.path.exists("joke-bot.pkl"):
    model = "babbage"
    # prompts = [f'Joke about {joke["category"]} : {joke["joke"]}' for joke in training_jokes.to_dict(orient="records")]

    if not os.path.exists(rf'../other/jokes/cleaned_compiled_data/fine_tune_dataset_prepared.jsonl'):
        print("Preparing data...")
        try:
            response = os.system(
                "openai tools fine_tunes.prepare_data -f ../other/jokes/cleaned_compiled_data/fine_tune_dataset.json")

        except:
            print("Error preparing data")

    print("Data prepared for fine tuning")
    print("-----------------------------")
    print("Fine tuning a new model...")
    try:
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY_1")
        os.system(
            f"openai api fine_tunes.create -t ../other/jokes/cleaned_compiled_data/fine_tune_dataset_prepared.jsonl -m {model}")
        os.system("openai api fine_tunes.follow -i ft-OmfrXmq7iXU5oaKEJE0qDW8t")
    except Exception as e:
        print("Error creating model: ")
        print(e)

    print("Training complete!")
