import pandas as pd
import numpy as np 
import split_dataset
import openai 
import json 

def main():

    df = pd.read_csv("data/raw/ds165_equations.txt", sep="\t")
    skill_col = df['KC (Default)']
    is_hint = df['First Attempt'] == 'hint'
    is_na = pd.isna(skill_col)
    is_control = df['Condition'] == 'control'

    df = df[~is_hint & ~is_na & is_control]
    
    df['new_step'] = make_step_col(df)

    problem_col = 'new_step'
    #problem_col = 'Problem Name'
    #problem_col = 'KC (Default)'

    print("Trials: %d" % df.shape[0])
    print("Students: %d" % pd.unique(df['Anon Student Id']).shape[0])
    print("Problems: %d" % pd.unique(df['Problem Name']).shape[0])
    print("Steps: %d" % pd.unique(df['new_step']).shape[0])
    print("Skills: %d" % (pd.unique(skill_col).shape[0]))
    
    problem_id, problem_text_to_id = to_numeric_sequence(df[problem_col])
    with open("data/datasets/equations.problem_text_to_id.json", "w") as f:
        json.dump(problem_text_to_id, f, indent=4)
    
    output_df = pd.DataFrame({
        "student" : to_numeric_sequence(df['Anon Student Id'])[0],
        "problem" : problem_id,
        "skill" : to_numeric_sequence(df['KC (Default)'])[0],
        "correct" : [1 if c == 'correct' else 0 for c in df['First Attempt']]
    })

    print(output_df)

    output_df.to_csv("data/datasets/equations.csv", index=False)
    full_splits = split_dataset.main(output_df)
    np.save("data/splits/equations.npy", full_splits)
    
    problem_id_to_text = { v: k for k, v in problem_text_to_id.items() }
    problems = [problem_id_to_text[i] for i in range(len(problem_id_to_text))]
    embeddings = embed_problems(problems)

    print(embeddings.shape)
    np.save("data/datasets/equations.embeddings.npy", embeddings)

def make_step_col(df):

    problems = df['Problem Name'].tolist()
    steps = df['Step Name'].tolist()

    rows = []
    curr_problem = None
    for i in range(df.shape[0]):
        if problems[i] != curr_problem:
            rows.append([])
            curr_problem = problems[i]
        rows[-1].append(steps[i])
    
    new_steps = []
    for row in rows:
        from_step = row[:-1]
        to_step = row[1:]
        ns = ["transform %s to %s" % pair for pair in zip(from_step, to_step)]
        ns += ['solve %s' % row[-1]]
        new_steps.extend(ns)
    
    return new_steps

def to_numeric_sequence(vals):

    sorted_vals = sorted(set(vals))

    mapping =  dict(zip(sorted_vals, range(len(vals))))

    return [mapping[v] for v in vals], mapping

def embed_problems(problems):

    problems = [p.strip() for p in problems]
    
    all_embeddings = []
    for i in range(0, len(problems), 500):
        embd = get_embeddings(problems[i:(i+500)])
        all_embeddings.append(embd)
    
    return np.vstack(all_embeddings)

def get_embeddings(question_texts, model="text-embedding-ada-002"):

    with open('openai_api_key', 'r') as f:
        api_key = f.read()
    
    openai.api_key = api_key
   
    results = openai.Embedding.create(input = question_texts, model=model)['data']
    return np.array([r['embedding'] for r in results])

def get_embeddings(question_texts, model="text-embedding-ada-002"):

    with open('openai_api_key', 'r') as f:
        api_key = f.read()
    
    openai.api_key = api_key
   
    results = openai.Embedding.create(input = question_texts, model=model)['data']
    return np.array([r['embedding'] for r in results])

if __name__ == "__main__":
    main()