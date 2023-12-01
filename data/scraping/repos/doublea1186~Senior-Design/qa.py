import openai
import os
import csv
import re

openai.api_key = "key"
os.environ['OPENAI_API_KEY']=openai.api_key

def read_file(input_file: str):
    with open(input_file, "r") as f:
        text_lines = f.readlines()
        return text_lines

def split_sentence_chunks(chunk_size: int, text_lines : list):
  new_lines = []
  new_text = ""
  for i, line in enumerate(text_lines):
    if i == (len(text_lines) - 1):
        if len(new_text) <= chunk_size:
           new_text += line
           new_lines.append(new_text)
    if len(new_text) > chunk_size:
      new_lines.append(new_text)
      new_text = ""
      new_text += line
    elif len(new_text + line) > chunk_size:
        new_lines.append(new_text)
        new_text = ""
        new_text += line
    else:
      new_text += line
  return new_lines[:-1]

def removing_new_lines(new_lines):
  res = []
  for sub in new_lines:
    res.append(sub.replace("\n", ""))
  return res

def get_questions(student_summary):
  prompt_2 = "StudentSummary:  " + student_summary + "\n"
  response = openai.Completion.create(
    model="davinci:ft-ccb-lab-members-2022-08-18-02-22-56",
    prompt=prompt_2,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["###"]
  )
  return response['choices'][0]['text']

def parse_completion(completion):
    questions = completion.strip().split('Question ')
    q_a_pairs = []
    for q in questions:
        q_a_pair = q.strip().split('Answer ')
        if len(q_a_pair) > 1:
            question = q_a_pair[0]
            answer = q_a_pair[1]

            question = re.sub('[0-9]. ', '', question)
            answer = re.sub('[0-9]. ', '', answer)
            q_a_pairs.append((question, answer))
    return q_a_pairs

def generate_questions_in_csv(output_file : str, summary_chunk_list : list, class_name : str, lecture_number : int):
    csv_fields = ['ClassName', 'LectureNumber', 'QuestionNumber','Explanation', 'Question', 'Answer'] 

    # data rows of csv file 
    rows = []

    q_number = 0
    for _, text in enumerate(summary_chunk_list):
        # print(i)
        completion = get_questions(text)
        q_a_pairs = parse_completion(completion)
        for qa in q_a_pairs:
            if len(qa[0]) > 3:
                rows.append([class_name, lecture_number, q_number, text, qa[0], qa[1]])
                q_number += 1

    # name of csv file 
    filename = output_file
            
    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
                
        # writing the fields 
        csvwriter.writerow(csv_fields) 
                
        # writing the data rows 
        csvwriter.writerows(rows)

def main_generate_questions(input_summary_file_name, output_qa_file_name, class_name, lecture_number):
  text_lines = read_file(input_summary_file_name)
  chunked_for_qa = split_sentence_chunks(500, text_lines)
  cleaned_for_qa = removing_new_lines(chunked_for_qa)
  generate_questions_in_csv(output_qa_file_name, cleaned_for_qa, class_name, lecture_number)

# if __name__ == '__main__':
#     input_file = "./resources/CIS521_L1_summary_2.txt"
#     output_file = "question_answer.csv"

#     text_lines = read_file(input_file)
#     chunked_for_qa = split_sentence_chunks(500, text_lines)
#     cleaned_for_qa = removing_new_lines(chunked_for_qa)
#     class_name = "AI and Philosophy"
#     lecture_number = 1
#     generate_questions_in_csv(output_file, cleaned_for_qa, class_name, lecture_number)


    
