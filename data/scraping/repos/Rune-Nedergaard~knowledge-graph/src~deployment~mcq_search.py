import os
import sys
import pandas as pd
import openai
from api_secrets import API_KEY
import re
from concurrent.futures import ThreadPoolExecutor
import concurrent

openai.api_key = API_KEY

import tiktoken


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.qa_search import find_question_answer_pairs, initialize_qa_search
from src.deployment.divide_mcq import divide_mcq
from src.deployment.process_subquestions import relevant_qa_pairs

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
def count_gpt35_tokens(text):
    tokens = tokenizer.encode_ordinary(text)
    return len(tokens)

def find_relevant_qa_pairs(mcq, df, sentence_model, question_embeddings, questions, k=10):
    subquestions = divide_mcq(mcq)
    relevant_indices = set()

    for count, subquestion in enumerate(subquestions):
        print(f"Behandler underspørgsmål {count+1}")
        subquestion_indices = relevant_qa_pairs(mcq, subquestion, df, sentence_model, question_embeddings, questions)
        relevant_indices.update(subquestion_indices)

    return relevant_indices, subquestions


def create_gpt4_prompt(question, relevant_qa_list):
    prompt = f"Multiple choice spørgsmål: {question}\n\n"
    prompt += "Spørgsmål-svar par:\n\n"
    for i, (qa_question, qa_answer, date) in enumerate(relevant_qa_list):#removed date, seemed less usefull
        prompt += f"{i + 1}:Spørgsmål: {qa_question}\nSvar: {qa_answer}\n\n"

    #check if prompt is above 6000 tokens using count function
    if count_gpt35_tokens(prompt) > 5500:
        #remove the last qa pair
        prompt = prompt.rsplit('\n\n', 2)[0]
        print("Prompt too long, removing last qa pair")
        while count_gpt35_tokens(prompt) > 5500:
            prompt = prompt.rsplit('\n\n', 1)[0]
            print("Prompt too long, removing last qa pair again...")
    return prompt

def get_gpt4_response(gpt4_prompt):
    print("Genererer svar baseret på relevante spørgsmål-svar par identificeret vha. søgning på underspørgsmål...")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": """Du besvarer multiple choice spørgsmål ud fra information i vedlagte spørgsmål-svar par. Du må kun tage udgangspunkt i informationen i disse spørgsmål-svar par og ikke benytte anden viden. Besvar multiple choice spørgsmålet og giv dernæst en forklaring på, hvorfor du har valgt det svar. Referer til mindst 5 relevante spørgsmål-svar par i din forklaring. Såfremt de vedlagte spørgsmål-svar ikke indeholder nok information til at besvares spørgsmålet, må du forklare at du gætter og eventuelt benytte hvad end de indeholder af små ledetråde til at vælge dit bedste gæt. Du skal svare, så hvis du intet ved må du gætte a/b/c/d helt tilfældigt og sige dette. Angiv dit svar således:

Svar: [a/b/c/d]

Forklaring [indsæt]

Indeks på relevante par: [3,7,8,..etc.]"""},
            {"role": "user", "content": gpt4_prompt},
        ],
        temperature=0.5,
        max_tokens=2000,
    )

    answer = response['choices'][0]['message']['content']
    return answer

def get_relevant_indices_from_response(gpt4_response, relevant_qa_list, df):
    relevant_pair_indices = re.findall(r"Indeks på relevante par: ?\[?([0-9,\s]+)\]?", gpt4_response)
    
    if not relevant_pair_indices:
        relevant_pair_indices = re.findall(r"Indeks på relevante par: ([0-9,\s]+)", gpt4_response)
    
    if relevant_pair_indices:
        try:
            relevant_pair_indices = [int(idx) - 1 for idx in re.split(r',\s*', relevant_pair_indices[0])]
        except:
            print("Kunne ikke finde relevante par i GPT-4 svar. Vælger de 5 første par.")
            relevant_pair_indices = list(range(5))
    else:
        print("Ingen relevante par fundet i GPT-4 svar. Vælger de 5 første par.")
        relevant_pair_indices = list(range(5))

    relevant_df_indices = []
    for idx in relevant_pair_indices:
        question, answer, date = relevant_qa_list[idx][:3]
        original_index = df[(df['Spørgsmål'] == question) & (df['Svar'] == answer)].index[0]
        relevant_df_indices.append(original_index)

    return relevant_df_indices, relevant_pair_indices


def read_mcq_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        mcq_list = content.strip().split('\n')
    return mcq_list

#necessary when processing fails for a few files for some reason
def get_existing_output_files(output_folder):
    files = [int(f.split('.')[0]) for f in os.listdir(output_folder) if f.endswith('.txt')]
    return set(files)



def process_mcq(count, mcq, df, sentence_model, question_embeddings, questions):
    print(f"Behandler følgende multiple choice spørgsmål: {mcq}\n")
    relevant_indices, subquestions = find_relevant_qa_pairs(mcq, df, sentence_model, question_embeddings, questions)
    relevant_qa_list = [(df.loc[idx, 'Spørgsmål'], df.loc[idx, 'Svar'], df.loc[idx, 'date']) for idx in relevant_indices]#removed titel
    gpt4_prompt = create_gpt4_prompt(mcq, relevant_qa_list)
    gpt4_response = get_gpt4_response(gpt4_prompt)
    print(f"Modellens svar:\n{gpt4_response}\n")

    # Find the relevant QA pair indices in the GPT-4 response
    relevant_df_indices, relevant_pair_indices = get_relevant_indices_from_response(gpt4_response, relevant_qa_list, df)

    # Save GPT-4 input and output to a txt file
    with open(f'data/mc_results/{count}.txt', 'w', encoding='utf-8') as f:
        # Write the MCQ
        f.write("Multiple choice spørgsmål:\n")
        f.write(mcq)
        #Write the underspørgsmål that MCQ was divided into
        f.write("\n")
        f.write("\nGenererede underspørgsmål:\n")
        for subquestion in subquestions:
            f.write(subquestion)
            f.write("\n")
        f.write("\nMultiple choice Svar:\n")
        f.write(gpt4_response)
        f.write("\n")
        # Write the relevant QA pairs
        f.write("\nIdentificerede relevante spørgsmål-svar par:\n\n")
        for (df_idx, qa_idx) in zip(relevant_df_indices, relevant_pair_indices):
            question, answer, date, titel, filurl = df.loc[df_idx, ['Spørgsmål', 'Svar', 'date', 'titel', 'filurl']]
            #write the index
            f.write(f"Par: {qa_idx+1}\n")
            f.write(f"Spørgsmål: {question}\nSvar: {answer}\nDato: {date}\nDokument titel:{titel}\nLink: {filurl}\n\n")
        f.close()




if __name__ == "__main__":
    file_path = 'data/videnstest.txt'
    mcq_list = read_mcq_file(file_path)

    # Get the existing output files
    output_folder = 'data/mc_results/'
    existing_output_files = get_existing_output_files(output_folder)

    # Filter the mcq_list to exclude existing output files
    filtered_mcq_list = [(idx + 1, mcq) for idx, mcq in enumerate(mcq_list) if idx + 1 not in existing_output_files]


    df, sentence_model, question_embeddings, questions, device = initialize_qa_search()

        # Create a ThreadPoolExecutor with a specific number of worker threads (e.g., 4)
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit the tasks to the ThreadPoolExecutor
        futures = [executor.submit(process_mcq, count, mcq, df, sentence_model, question_embeddings, questions) for count, mcq in filtered_mcq_list]

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            pass