from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from datetime import datetime
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def get_api():
    key = os.environ["ANTHROPIC_API_KEY"]
    anthropic = Anthropic(
        api_key=key
    )
    return anthropic
claude = get_api()

def get_prompt_templates():
    file = open(r"prompt_03_brief", "r")
    file02 = open(r"prompt_04_predictions", "r")
    file03 = open(r"prompt_05_suggestions", "r")
    file04 = open(r"prompt_06_simulation", "r")
    file05 = open(r"prompt_07_finalOutput", "r")
    pt_brief = file.read()
    pt_predictions = file02.read()
    pt_suggestions = file03.read()
    pt_simulation = file04.read()
    pt_finalOutput = file05.read()

    return pt_brief, pt_predictions, pt_suggestions, pt_simulation, pt_finalOutput

pt_brief, pt_predictions, pt_suggestions, pt_simulation, pt_finalOutput = get_prompt_templates()

def get_history():
    if os.path.exists('user_events.csv'):
        df = pd.read_csv('user_events.csv')
    else:
        df = pd.DataFrame(columns=['date', 'situation'])
    return df
df = get_history()

def check_env_var(env_var):
    if env_var in os.environ:
        print('ANTHROPIC_API_KEY is set!')
    else:
        print('ANTHROPIC_API_KEY is not set!')
check_env_var('ANTHROPIC_API_KEY')

situation = input("Enter your situation: ")
now = datetime.now()
new_row = pd.DataFrame([[now, situation]], columns=['date', 'situation'])
df = pd.concat([df, new_row], ignore_index=True)

nlp = spacy.load('en_core_web_lg')
def get_context(user_input, data):
    df = pd.read_csv(data)

    # Add vector column
    df['Vector'] = df['situation'].apply(lambda x: nlp(x).vector)

    # Get user vector
    user_vector = nlp(user_input).vector

    # Compute similarities
    sims = df['Vector'].apply(lambda x: cosine_similarity([x], [user_vector]))
    df['Similarity'] = sims

    # Sort by similarity
    df = df.sort_values('Similarity', ascending=False)

    # Get top 3 situations
    top3 = df[['situation', 'date']].head(3)

    return top3.values.tolist()

if os.path.exists('user_events.csv'):
    embedding_results = get_context(situation, "user_events.csv")
else:
    embedding_results = []

def list_to_string(list):
  string = ""
  for item in list:
    string += str(item) + "\n"
  return string

context_lines = list_to_string(embedding_results)

# Save dataframe
df.to_csv('user_events.csv', index=False)

situation_prompt = "[situation]:" + situation + "\n"
context = "[context]:" + context_lines + "\n"
thoughts = "[thoughts]:" + input("What is on your mind?") + "\n"
info_lookup = "[info_lookup]:" + "Current date is " + str(now) + "\n"
# advices_lookup = "[advices_lookup]:" + advices + "\n"
end_prompt_pt_brief = f"{HUMAN_PROMPT}" + pt_brief + thoughts + situation_prompt + info_lookup + context + f"{AI_PROMPT}"

def chat_completion(a,model_input):
    completion = claude.completions.create(
        model=model_input,
        max_tokens_to_sample=100000,
        temperature=0,
        prompt=a,
    )
    return completion
print("Retrieving histories and references...")
response = chat_completion(end_prompt_pt_brief, "claude-1")
pt_brief_response = "[summary]:" + str(response.completion) + "\n"
# print(pt_brief_response)

def get_pt_predictions(pt_predictions, pt_brief_response):
    end_prompt_pt_predictions = f"{HUMAN_PROMPT}" + pt_predictions + "\n" + pt_brief_response + f"{AI_PROMPT}"
    return end_prompt_pt_predictions
print("Predicting potential scenarios...")
end_prompt_pt_predictions = get_pt_predictions(pt_predictions, pt_brief_response)
pt_predictions_response = chat_completion(end_prompt_pt_predictions, "claude-2").completion
# print(pt_predictions_response)

def parse_scenarios(data):
    parsed = {}

    if isinstance(data, str):
        # handle string input
        for line in data.split('\n'):
            if re.match(r'\d+\.', line):
                number = int(line[0])
                scenario = line.split('. ')[1]
                parsed[number] = scenario

    elif isinstance(data, dict):
        # handle dict input
        for key, value in data.items():
            match = re.search(r'(\d+)\. (.*)', value['scenario'])
            if match:
                number = int(match.group(1))
                scenario = match.group(2)
                parsed[number] = scenario

    return parsed
def get_scenario_responses(text):
    scenarios = parse_scenarios(text)

    responses = {}

    for i, scenario in scenarios.items():
        scenario_prompt = f"{HUMAN_PROMPT}" + pt_suggestions + "\n" + "[predictions]:" + scenario + "\n" + pt_brief_response + "[thoughts]:" + thoughts + "\n" + f"{AI_PROMPT}"
        response = chat_completion(scenario_prompt, "claude-2")

        responses[i] = {
            "scenario": scenario,
            "response": response.completion
        }

    return responses
print("Computing suggestions...")
suggestions_response = get_scenario_responses(pt_predictions_response)

if len(suggestions_response) <= 0:
    suggestions_dict = get_scenario_responses(suggestions_response)
else:
    print(suggestions_response)

def parse_suggestions(data):
    parsed = {}

    for id, scenario in data.items():

        response = scenario['response']
        suggestions = []
        extracted = set()

        for i in range(5):

            match = re.search(r'\d\. (.*)', response)
            if match:
                text = match.group(1)
                if text not in extracted:
                    extracted.add(text)
                    suggestions.append(text)

                # Update search to start after current match
                response = response[match.end():]

        parsed[id] = suggestions

    return parsed
suggestions_dict = parse_suggestions(suggestions_response)

if len(suggestions_dict) <= 0:
    suggestions_dict = parse_suggestions(suggestions_response)
else:
    print(suggestions_dict)

def add_scores(data):
    new_data = {}

    for level in data:

        advice_list = data.get(level, [])

        if len(advice_list) > 0:

            new_advice_list = []

            for advice in advice_list:

                prompt = f"{HUMAN_PROMPT}" + pt_simulation + "\n" + "[suggested_next_move]:" + advice + "\n" + pt_brief_response + "[thoughts]:" + thoughts + "\n" + f"{AI_PROMPT}"
                response = chat_completion(prompt, "claude-1").completion

                # print response for debugging
                print(response)

                # use regex to extract score
                match = re.search(r'Recommendation Score = (\d+)', response)
                if match:
                    score = match.group(1)
                else:
                    score = "N/A"

                # handle invalid score
                if score == "N/A":
                    print(f"Could not extract score from response: {response}")

                # add scored advice
                new_advice = f"{advice} (recommendation: {score}%)"
                new_advice_list.append(new_advice)

            new_data[level] = new_advice_list

    return new_data
print("Computing feasibilities...")
output_dict = add_scores(suggestions_dict)
# print(output_dict)

def get_top_advice(data):
    top_list = []

    for level in data:
        advice_list = data[level]
        for advice in advice_list:
            if "%" in advice:
                score_str = advice.split("(recommendation: ")[-1][:-2]
                if score_str != "N/A":
                    score = int(score_str)
                    if score > 80:
                        top_list.append(advice)

    if not top_list:
        top_list = []
        for advice_list in data.values():
            if advice_list:
                top_advice = advice_list[0]
                top_list.append(top_advice)

    return top_list[:3]
top = get_top_advice(output_dict)
# print(top)

def count_top_advice(top_advice):

  num_entries = len(top_advice)

  print(f"Number of top advice entries: {num_entries}")

  return num_entries
num_entries = count_top_advice(top)

def cluster_top_advice(advice_list, n_clusters=3):
    # extract text from advice strings
    advice_text = [text.split("(recommendation:")[0].strip() for text in advice_list]

    # vectorize text using TF-IDF
    vectorizer = TfidfVectorizer()
    advice_vectors = vectorizer.fit_transform(advice_text)

    # cluster advice vectors
    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit_predict(advice_vectors)

    # map clusters back to original advice
    advice_with_cluster = dict(zip(advice_list, clusters))

    # take top 3 clusters by size
    sorted_clusters = sorted(set(list(clusters)), key=list(clusters).count, reverse=True)[:3]

    # return representative advice for each cluster
    return [list(advice_with_cluster.keys())[list(advice_with_cluster.values()).index(c)] for c in sorted_clusters]

def interactive_advice(top_advice):
    print("Top Advice Options:")
    for i, advice in enumerate(top_advice):
        print(f"{i + 1}. {advice}")

    print("0. End query")

    while True:
        selection = input("Select advice to elaborate (1-%d): " % len(top_advice))

        if selection == '0':
            print("Query ended.")
            break

        idx = int(selection) - 1
        if idx < 0 or idx >= len(top_advice):
            print("Invalid selection")
            continue

        selected = top_advice[idx]

        print()
        print("You selected:", selected)
        prompt = f"{HUMAN_PROMPT}" + pt_finalOutput + "\n" + "[summary]:" + pt_brief_response + "[suggestion]:" + selected + "[thoughts]:" + thoughts + "\n" + f"{AI_PROMPT}"
        elaboration = chat_completion(prompt, "claude-2").completion
        print(elaboration)

        print()

    print("Thank you for using the interactive advice system!")
print("Done!")
if num_entries > 3:
    top3 = cluster_top_advice(top, n_clusters=3)
    interactive_advice(top3)
else:
    interactive_advice(top)
