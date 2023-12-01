import random
from datetime import datetime, timezone
import pandas as pd
from airflow.models import Variable
import openai
import Levenshtein


def generate_prompt(target_vocab, selection_vocabularies):
    prompt = f"Please generate me a very basic vocabulary example for '{target_vocab}', considering the selection vocabs below. \
Also please give me a clear example that will only match with the target vocabulary and not the selection vocabularies listed below. \
Also, please make sure you use exactly the word '{target_vocab}' in the example and output a single sentence.\n\n"
    prompt += "Selection vocabs:\n"
    for i, selection_vocab in enumerate(selection_vocabularies):
        prompt += f"{i+1}. '{selection_vocab}'\n"

    prompt += "\nExample for the target vocab:"

    return prompt

def choose_best_example(target_vocab, selection_vocabularies):
    prompt = generate_prompt(target_vocab, selection_vocabularies)
    openai.api_key = Variable.get("openai_token")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},],
        temperature=0.0,
    )
    content = response["choices"][0]["message"]["content"]

    return content


def find_best_match(keyword, word_list):
    best_match = None
    best_distance = float('inf')

    if len(keyword.split()) == 1:
        for word in word_list:
            distance = Levenshtein.distance(keyword, word.lower())
            if distance < best_distance:
                best_match = word
                best_distance = distance
    else:
        for i in range(len(word_list) - len(keyword.split()) + 1):
            phrase = " ".join(word_list[i:i + len(keyword.split())])
            distance = Levenshtein.distance(keyword, phrase.lower())
            if distance < best_distance:
                best_match = phrase
                best_distance = distance

    return best_match

def send_slack_quiz(vocab_df, quiz_details_df, user_id, target_lang, con):

    # Set the quiz target
    quiz_target = vocab_df[vocab_df['quizzed_count'] == vocab_df['quizzed_count'].min()]

    # mix quiz_target
    quiz_target = quiz_target.sample(n=1)
    quiz_target = pd.DataFrame(quiz_target)

    # Get 5 random vocabularies that have exposure greater than 3
    selection_df = vocab_df[vocab_df['exposure'] >= 3]
    selection_df = selection_df[selection_df['vocab'] != quiz_target['vocab'].values[0]]

    if len(selection_df["vocab"]) < 5:
        print("Not enough vocabularies to generate quiz")
        return None, vocab_df

    selection_vocabs = selection_df["vocab"].sample(n=4)
    target_vocab = quiz_target["vocab"].values[0]
    target_vocab_id = quiz_target["vocab_id"].values[0]

    # Generate prompt  
    prompt = generate_prompt(target_vocab, selection_vocabs)

    # Get best example using openai API
    best_example = choose_best_example(target_vocab, selection_vocabs)

    # Locate the target_vocab in the selection_vocabs and replace it with "_________"
    best_example = "Q: " + best_example.replace(target_vocab.capitalize(), "`_________`")
    best_example = best_example.replace(target_vocab.lower(), "`_________`")

    if "_________" not in best_example:
        best_match = find_best_match(target_vocab, best_example.split(" "))
        best_example = best_example.replace(best_match, "`_________`")

    # Form the slack json message
    options = []
    selection_vocabs = list(selection_vocabs)
    selection_vocabs += [target_vocab]
    random.shuffle(selection_vocabs)  # mix selection_vocabs
    for ind, selection in enumerate(selection_vocabs):
        selection_jsn = {
                            "text": {
                                "type": "plain_text",
                                "text": selection,
                                "emoji": True
                            },
                            "value": f"value-{ind + 1}"
                        }
        options.append(selection_jsn)
    
    blocks = [
            {"type": "divider"},
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Vocabulary Quiz ✨",
                }
            },
            {
                "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{best_example}*"
                    }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Please select the correct answer from the options below:"
                }
            },
            {
                "type": "input",
                "element": {
                    "type": "static_select",
                    "placeholder": {
                        "type": "plain_text",
                        "text": "Select a vocab",
                        "emoji": True
                    },
                    "options": options,
                    "action_id": target_vocab_id
                },
                "label": {
                    "type": "plain_text",
                    "text": "Select Answer",
                    "emoji": True
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Submit",
                            "emoji": True
                        },
                        "value": "submit_button",
                        "action_id": "button_click",
                        "style": "primary"
                    }
                ]
            }
        ]

    user_id = quiz_target["user_id"].iloc[0]
    vocab_id = quiz_target["vocab_id"].iloc[0]
    target_vocab = quiz_target["vocab"].iloc[0]
    # Randomly generate a quiz_id
    quiz_id = vocab_id + user_id

    # Combine vocab_id and user_id to form a unique id
    quiz_tbl = pd.DataFrame({"quiz_id": quiz_id,
                             "user_id": user_id,
                             "vocab_id": vocab_id,
                             "target_vocab": target_vocab,
                             "selected_vocab": None,
                             "quiz_content": best_example,
                             "quizzed_at_utc": datetime.now(timezone.utc),
                             "quiz_submitted_at_utc": None,
                             "status": "quiz_sent"}, index=[0])
    
    vocab_df.loc[vocab_df["vocab_id"] == target_vocab_id, "quizzed_count"] += 1
    quiz_tbl.to_sql("quiz_details", con, if_exists="append", index=False)
    return blocks, vocab_df