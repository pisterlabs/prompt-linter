import json

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

import argparse
from document import Report
from reader import Reader
import os

ORIGINAL_PROMPT = """As a senior equity analyst with expertise in climate science evaluating a company's sustainability report, you are presented with the following background information:

{basic information}

With the above information and the following extracted components of the sustainability report at hand, please respond to the posed question, ensuring to reference the relevant parts ("SOURCES").
Format your answer in JSON format with the two keys: ANSWER (this should contain your answer string without sources), and SOURCES (this should be a list of the source numbers that were referenced in your answer).

QUESTION: {question}
=========
{extracted relevant information from the report}
=========
"""

PROMPT_BEGINNING = """
As a senior equity analyst with expertise in climate science evaluating a company's sustainability report, you are presented with the following background information:

{basic_info}

With the above information and the following extracted components of the sustainability report at hand, please respond to the posed question, ensuring to reference the relevant parts ("SOURCES").
Format your answer in JSON format with the two keys: ANSWER (this should contain your answer string without sources), and SOURCES (this should be a list of the source numbers that were referenced in your answer).

QUESTION: {question}
=========
{summaries}
=========
"""

GUIDELINE_LENGTH = 8
GUIDELINE_LIST = """Please adhere to the following guidelines in your answer:
1. Your response must be precise, thorough, and grounded on specific extracts from the report to verify its authenticity.
2. If certain information is unclear or unavailable, admit the lack of knowledge rather than devising an answer.
3. Answer the question strictly based on the provided extracts. If the information available is insufficient, clearly state that the question cannot be answered based on the given report.
4. Keep your ANSWER within {answer_length} words.
5. Be skeptical to the information disclosed in the report as there might be greenwashing (exaggerating the firm's environmental responsibility). Always answer in a critical tone.
6. cheap talks are statements that are costless to make and may not necessarily reflect the true intentions or future actions of the company. Be critical for all cheap talks you discovered in the report.
7. Always acknowledge that the information provided is representing the company's view based on its report.
8. Scrutinize whether the report is grounded in quantifiable, concrete data or vague, unverifiable statements, and communicate your findings.
"""

PROMPT_ENDING = """Your FINAL_ANSWER in JSON (ensure there's no format error):
"""


REFINE_PROMPT = """
You are a prompt engineer improving <Previous Prompt> given <Expert Feedback> and <AI's Previous Response>.

1. <Previous Prompt>: \"\"\"{original_prompt}

<Old Guideline List>: {guideline_list}
\"\"\"

2. <AI’s Previous Response>: \"\"\"{old_response}\"\"\"

3. <Expert Feedback>: "{feedback}"

Given this feedback, could you please generate a new guideline that we can add to our existing list (<Old Guideline List>) to enhance future outputs, making sure to only include the requested details if such information exists in the report?

The new guideline should be concise and easy to follow by an AI assistant. Please format your answer in JSON with a single key “GUIDELINE”

Your answer in JSON (make sure there’s no format error):"""


def find_answer(string, name="GUIDELINE"):
        for l in string.split('\n'):
            if name in l:
                start = l.find(":") + 3
                end = len(l) - 1
                return l[start:end]
        return string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, default='On-Impact-Progress-Report-2021.pdf')
    parser.add_argument("--basic_info_dir", type=str, default='data/basic_info')
    parser.add_argument("--answers_dir", type=str, default='data/answers')
    parser.add_argument("--assessment_dir", type=str, default='data/assessment')
    parser.add_argument("--vector_db_dir", type=str, default='data/vector_db')
    parser.add_argument("--user_qa_dir", type=str, default='data/user_qa')
    parser.add_argument("--prompt_eng_dir", type=str, default='data/auto_prompt_eng')
    parser.add_argument("--user_question", type=str, default='')
    parser.add_argument("--answer_length", type=int, default=50)
    parser.add_argument("--detail", action='store_true', default=False)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    assert (args.user_question != '')

    report_name = args.pdf_path.split('/')[-1]
    assert report_name.endswith('.pdf')
    report_name = report_name.replace('.pdf', '')
    user_question_file = report_name + '_user_qa.jsonl'

    if not os.path.exists(args.basic_info_dir):
        os.makedirs(args.basic_info_dir)
    if not os.path.exists(args.answers_dir):
        os.makedirs(args.answers_dir)
    if not os.path.exists(args.assessment_dir):
        os.makedirs(args.assessment_dir)
    if not os.path.exists(args.vector_db_dir):
        os.makedirs(args.vector_db_dir)
    if not os.path.exists(args.user_qa_dir):
        os.makedirs(args.user_qa_dir)
    if not os.path.exists(args.prompt_eng_dir):
        os.makedirs(args.prompt_eng_dir)
    destination_folder = "data/pdf/"
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    report = Report(
        path=args.pdf_path,
        store_path=os.path.join(destination_folder, args.pdf_path.split('/')[-1]),
        db_path=os.path.join(args.vector_db_dir, report_name),
    )

    reader = Reader(llm_name="gpt-3.5-turbo-16k", answer_length=str(args.answer_length))
    engineering_template = PromptTemplate(template=REFINE_PROMPT, input_variables=["original_prompt", "guideline_list", "old_response", "feedback"])
    print("=====Starting Automatic Prompt Engineering=====")
    while True:
        output, _ = reader.user_qa(args.user_question, report,
                                   basic_info_path=os.path.join(args.basic_info_dir, report_name + '.json'),
                                   answer_length=args.answer_length,
                                   prompt_template=PROMPT_BEGINNING + '\n' + GUIDELINE_LIST + '\n' + PROMPT_ENDING,
                                   various_source=args.detail, top_k=args.top_k)
        answer = {"ANSWER": output['ANSWER'], "SOURCES": output['SOURCES']}
        print("\nGiven your question and the current prompt, the answer is:")
        print(answer)
        print('PAGE: ', output['PAGE'])
        print("\n\nPlease input your suggestion/feedback to the current answer here. Your suggestion will be used to improve the prompt:")
        expert_feedback = input()
        current_prompt = engineering_template.format(original_prompt=ORIGINAL_PROMPT, guideline_list=GUIDELINE_LIST, old_response=answer, feedback=expert_feedback)
        message = [
            SystemMessage(content="You are a helpful prompt engineer."),
            HumanMessage(content=current_prompt)
        ]
        llm = ChatOpenAI(temperature=0, max_tokens=256)
        output_text = llm(message).content
        try:
            output_dict = json.loads(output_text)
            if 'GUIDELINE' not in output_dict.keys():
                raise ValueError("Key name(s) not defined!")
        except Exception as e:
            output_dict = {'GUIDELINE': find_answer(output_text)}
        print("\nYour feedback is transferred to the following guideline:")
        new_guideline = str(GUIDELINE_LENGTH + 1) + '. ' + output_dict['GUIDELINE']
        print(new_guideline)
        with open(os.path.join(args.prompt_eng_dir, report_name + '.jsonl'), 'a') as f:
            prompt_json = json.dumps({"question": args.user_question, "feedback": expert_feedback, "guideline": new_guideline, "old_anwser": answer})
            f.write(prompt_json + '\n')
        GUIDELINE_LENGTH += 1
        GUIDELINE_LIST += new_guideline + '\n'






