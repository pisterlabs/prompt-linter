import logging
import openai
import time
import re
import json
import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)
MAX_API_RETRY = 5


def get_openai_response(sys_prompt, user_prompt, max_tokens, model="gpt-3.5-turbo") -> (str, int):
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            content = response["choices"][0]["message"]["content"]
            token = response["usage"]["total_tokens"]
            logger.info(content)
            return content, token
        except Exception as e:
            logger.error(e)
            time.sleep(5)
    logger.error(f"Failed after {MAX_API_RETRY} retries.")
    return "error", 0


class Evaluator:
    """
    Class for evaluating questions using the GPT-3.5 API.

    Attributes:

    """

    def __init__(self, output_dir='Eval/'):
        self.max_tokens = 1024
        self._generate_prompt()

    def _generate_prompt(self):
        self.prompt_template = """
        [Question]
        {question}

        [The Start of Assistant's Answer]
        {answer}

        [The End of Assistant's Answer]

        [The Start of Reference Answer]
        {reference}

        [The End of Reference Answer]

        [System]
        {prompt}

        """
        self.system_prompt = "You are an evaluator assessing the quality of the AI assistant's answer based solely on the retrieved context."
        self.prompt = """
Your task is to provide feedback on the performance of a single AI assistant in response to the user question displayed above. Please evaluate the assistant's response with the provided reference answer, which is entirely based on correctly retrieved information, in terms of its helpfulness, relevance, accuracy, and level of detail.
When evaluating, please assume that the assistant has retrieved the necessary information to respond.
The assistant's performance will be rated on a scale of 1 to 10, with a higher score indicating superior overall performance.Please output a very simple explanation first.

At the end, start a new line and output a single line containing only one value, indicating the score for the assistant, make sure you give a stable and objective scoring.
"""

    @staticmethod
    def read_jsonl_and_extract_fields(file_path, fields=None):
        # Function to read a jsonl file and extract specific fields
        data = []
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            json_line = json.loads(line)
            if fields:
                extracted_data = {field: json_line.get(field, None) for field in fields}
            else:
                extracted_data = json_line  # If no specific fields are provided, return all fields
            data.append(extracted_data)

        return data

    def pair_qa_infos(self, question_file, answer_file):
        # Extract all data from the question file
        question_data = self.read_jsonl_and_extract_fields(question_file)

        # Convert the extracted data to a dictionary for easy lookup by 'question_id'
        question_dict = {item['question_id']: item for item in question_data}

        # Extract all data from the answer file
        answer_data = self.read_jsonl_and_extract_fields(answer_file)

        # Pair the questions with the answers using 'question_id'
        qa_infos_pairs = []
        for item in answer_data:
            question_id = item['question_id']
            question_data = question_dict.get(question_id, None)

            if question_data:
                # The extracted question and reference_answer
                question = question_data.get('question')
                reference_answer = question_data.get('answer')

                # Extract other data fields as metadata
                metadata_fields = set(list(question_data.keys()) + list(item.keys())) - {'question_id', 'question', 'answer'}
                metadata = {field: item.get(field, None) for field in metadata_fields}
                metadata.update({field: question_data.get(field, None) for field in metadata_fields})

                qa_infos_pairs.append({
                    "question_id": question_id,
                    "question": question,
                    "reference_answer": reference_answer,
                    "answer": item.get('answer'),
                    "metadata": metadata
                })

        return qa_infos_pairs

    @staticmethod
    def _extract_score(review):
        scores = re.findall(r"(\d+(\.\d+)?)", review)
        if scores:
            score = float(scores[-1][0])
        else:
            score = None
        return score

    def evaluate(self, question_file_path, answer_file_path) -> List[Dict]:
        # Pairing the questions and answers
        qa_pairs = self.pair_qa_infos(question_file_path, answer_file_path)

        total_token_usage = 0
        review_jsons = []
        for qa_pair in qa_pairs:
            user_prompt = self.prompt_template.format(question=qa_pair['question'],
                                                      answer=qa_pair['answer'],
                                                      reference=qa_pair['reference_answer'],
                                                      prompt=self.prompt
                                                      )

            logger.info("Prompt Loaded")

            logger.info("Retrieving Response")
            review, token = get_openai_response(self.system_prompt, user_prompt, self.max_tokens)
            score = self._extract_score(review)
            total_token_usage += token
            review_jsons.append({
                "score": score,
                "question_id": qa_pair["question_id"],
                "metadata": qa_pair["metadata"],
                "judgement": review
            })

        logger.info("total_token_usage" + str(total_token_usage))
        return review_jsons

    def evaluate_to_jsonl(self, question_file_path, answer_file_path):
        review_jsons = self.evaluate(question_file_path, answer_file_path)

        date = datetime.datetime.now()
        reviews_file_path = f"reviews_{date}.jsonl".format(date=date.strftime("%Y-%m-%d"))
        file_path = self.output_dir + '/' + reviews_file_path

        with open(file_path, 'a') as file:
            for review_json in review_jsons:
                file.write(json.dumps(review_json) + '\n')

        return reviews_file_path
