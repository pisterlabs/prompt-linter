import openai
import pandas as pd

input_statements = pd.read_csv('DirectStatements.csv', header=None)[0].tolist()[7:]

print(len(input_statements))

openai.api_key = ""

def generate_softened_statement(statement):
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Please help rephrase the following statement into a question that gently encourages a person to consider their own perspective on their habits. The rephrased question should be open-ended, imply no judgment, and convey understanding and empathy."},
            {"role": "user", "content": statement}
        ]
    )
    return(completion.choices[0].message)
    

# Initialize a list to store the results
results = []

for statement in input_statements:
    output = generate_softened_statement(statement)
    meets_criteria = "Manual Review Needed"
    results.append([statement, output, meets_criteria])

# Create a DataFrame and save it to a CSV file
results_df = pd.DataFrame(results, columns=['Input Statement', 'Produced Output', 'Meets Criterion'])
results_df.to_csv('A4_4_4.csv', index=False)