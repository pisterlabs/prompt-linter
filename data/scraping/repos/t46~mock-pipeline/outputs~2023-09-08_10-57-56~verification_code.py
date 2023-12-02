from openai import OpenAI
from collections import Counter

# Define the experiment
general_prompts = ["What is 1 + 1?", "What is the square root of 4?", "What is 2 * 2?"]
specific_prompts = ["Provide the numerical answer to 1 + 1", "Provide the numerical answer to the square root of 4", "Provide the numerical answer to 2 * 2"]

# Initialize OpenAI
openai = OpenAI()

# Conduct the experiment
def conduct_experiment(prompts):
    responses = []
    for prompt in prompts:
        response = openai.Completion.create(engine="davinci-codex", prompt=prompt, max_tokens=5)
        responses.append(response.choices[0].text.strip())
    return responses

general_responses = conduct_experiment(general_prompts)
specific_responses = conduct_experiment(specific_prompts)

# Analyze the results
def analyze_responses(responses):
    categorized_responses = {"numerical only": [], "non-numerical": []}
    for response in responses:
        if response.isdigit():
            categorized_responses["numerical only"].append(response)
        else:
            categorized_responses["non-numerical"].append(response)
    return categorized_responses

general_categorized_responses = analyze_responses(general_responses)
specific_categorized_responses = analyze_responses(specific_responses)

general_proportion = len(general_categorized_responses["numerical only"]) / len(general_responses)
specific_proportion = len(specific_categorized_responses["numerical only"]) / len(specific_responses)

# Test the hypothesis
hypothesis_supported = specific_proportion > general_proportion

# Document the results
report = f"""
Experiment Report:

General Prompts: {general_prompts}
Specific Prompts: {specific_prompts}

General Responses: {general_responses}
Specific Responses: {specific_responses}

Categorized General Responses: {general_categorized_responses}
Categorized Specific Responses: {specific_categorized_responses}

Proportion of 'numerical only' responses for General Prompts: {general_proportion}
Proportion of 'numerical only' responses for Specific Prompts: {specific_proportion}

Hypothesis Supported: {hypothesis_supported}
"""

print(report)

# Review and refine
if not hypothesis_supported:
    print("The hypothesis was not supported. Consider refining the prompts or the categorization method.")
else:
    print("The hypothesis was supported. Consider further experiments to refine the understanding of how prompt specificity influences LLM responses.")