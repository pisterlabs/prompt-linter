from typing import Callable, List, Tuple

import numpy as np
import openai
from tqdm import tqdm

from api_key import api_key
from dataset.cosmos_qa import CosmosQA
from dataset.dataset import Dataset
from dataset.hellaswag import Hellaswag
from dataset.race import Race
from method.binary_method import BinaryMethod
from models import curie_finetune_hellaswag, curie_finetune_cosmos_qa_binary, curie_finetune_race_binary, \
    curie_finetune_hellaswag_binary
from method.natural_method import NaturalMethod


def evaluate(dataset: Dataset, predict: Callable) -> Tuple[float, List[int]]:
    questions = dataset.to_questions()
    print("Converted to list of Questions")
    result = []
    count = 0
    for question in tqdm(questions):
        pred = predict(question)
        result.append(pred)
        if pred == question.answer_idx:
            count += 1

    return count / len(questions), result


if __name__ == '__main__':
    # val = CosmosQA(split='validation')
    # val = Race('middle', split='validation')
    val = Hellaswag(split='validation')
    openai.api_key = api_key
    accuracy, answers = evaluate(val, lambda question: BinaryMethod.ask(question, model=curie_finetune_hellaswag_binary))

    np.savetxt('hellaswag_result_curie_binary.txt', X=np.array(answers))
    print(f'accuracy={accuracy}')
