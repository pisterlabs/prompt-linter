import os
import json

from openai_api import QueryAgent

EXP_NAME = "final-predictions"

if __name__ == "__main__":

    cur_path = os.path.abspath(os.curdir)

    # Retrieve the API key saved in environment
    api_key = os.getenv("OPENAI_API_KEY")

    # Instatiate the Query Agent for OpenAI API Calls
    with open(cur_path + "/configs/asqa.json", 'r') as f:

        # We use gpt-3.5-turbo-instruct for cost reduction (Input: $0.0015 / 1K tokens  Output: $0.0020 / 1K tokens) instead of ($0.0200 / 1K tokens)
        qa = QueryAgent(
            model='gpt-3.5-turbo-instruct',
            retrieval_kwargs=json.load(f),
            api_key=api_key, 
        )

    # Retrieve the evaluation questions
    with open(cur_path + "/dataset/ASQA.json", 'r') as f:

        questions = json.load(f)
        questions = questions['dev']

    assert(len(questions) == 500)

    # Gather predictions
    batch_size = 20
    batch_num = 0
    count = 0
    predictions = dict()
    print(f"Batch Size: {batch_size}")

    batch_keys = []
    batch_questions = []
    batch_responses = []
    
    for k, v in questions.items():

        batch_keys.append(k)
        batch_questions.append(v['ambiguous_question'])

        count += 1

        if count == batch_size:
            # Track batch num
            batch_num += 1
            print(f"Batch {batch_num} / {500/batch_size}")

            # Query the agent
            batch_responses = qa.respond(batch_questions)

            # Save the results
            for i in range(batch_size):
                predictions[batch_keys[i]] = batch_responses[i]

            # Reset batch variables
            batch_keys = []
            batch_questions = []
            batch_responses = []
            count = 0

    with open(cur_path + f"/outputs/{EXP_NAME}.json", 'w') as f:

        json.dump(predictions, f)
    
