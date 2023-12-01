import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import uuid, wandb
from langchain.prompts import PromptTemplate

import config as cfg

class QAEvaluation:
    def __init__(self, file_paths=cfg.EVA_PATH_CSV, model_name=cfg.MODEL_EVA):
        self.file_paths = file_paths
        self.model_name = model_name
        self.project_name = cfg.PROJECT_NAME
        self.generated_uuid = str(uuid.uuid4())
        self.dataset = None
        self.tokenizer = None
        self.model = None
        self.template = cfg.PROMPT_TEMP
        self.prompt = PromptTemplate(template=self.template, input_variables=[
            'prompt', 'a', 'b', 'c', 'd', 'e'
        ])
        self.aps = []

    def load_data(self):
        self.dataset = load_dataset("csv", data_files=[self.file_paths])

    def format_text(self, example):
        text = self.prompt.format(prompt=example['prompt'], a=example['A'], b=example['B'], c=example['C'], d=example['D'], e=example['E'])
        return {"text": text}

    def preprocess_data(self):
        self.dataset = self.dataset.map(self.format_text)

    def setup_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def get_ans(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        logits = self.model(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda()).logits[0, -1]

        # Create a list of tuples having (logit, 'option') format
        options_list = [(logits[self.tokenizer(' A').input_ids[-1]], 'A'), (logits[self.tokenizer(' B').input_ids[-1]], 'B'), (logits[self.tokenizer(' C').input_ids[-1]], 'C'), (logits[self.tokenizer(' D').input_ids[-1]], 'D'), (logits[self.tokenizer(' E').input_ids[-1]], 'E')]
        options_list = sorted(options_list, reverse=True)
        ans_list = []
        for i in range(3):
            ans_list.append(options_list[i][1])

        return ans_list

    @staticmethod
    def apk(actual, predicted, k=10):
        """
        Computes the average precision at k.

        This function computes the average prescision at k between two lists of
        items.

        Parameters
        ----------
        actual : list
                A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements

        Returns
        -------
        score : double
                The average precision at k over the input lists
        ref: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
        """
        if len(predicted)>k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)


    def evaluate(self):
        run = wandb.init(
            project=self.project_name,
            name=f'{self.model_name}-{self.generated_uuid}-evaluation',
            anonymous='must')

        eval_table =  wandb.Table(columns=[
            'prompt', 
            'answer', 
            'prediction-1', 
            'prediction-2', 
            'prediction-3', 
            'mAP', 
            'A', 
            'B', 
            'C', 
            'D', 
            'E'
            ])

        bar = tqdm(enumerate(self.dataset['train']), total=len(self.dataset['train']))
        for i, data in bar:
            ans_list = self.get_ans(data['text'])
            average_precision = self.apk([data['answer']], ans_list, k=3)
            self.aps.append(average_precision)
            ans1, ans2, ans3 = ans_list
            eval_table.add_data(data['prompt'],
                                data['answer'],
                                ans1,
                                ans2,
                                ans3,
                                average_precision,
                                data['A'],
                                data['B'],
                                data['C'],
                                data['D'],
                                data['E']
                                )

        wandb.log({'Evaluation': eval_table})
        run.finish()

        mAP = np.mean(self.aps)
        print(mAP)
        return mAP

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.setup_model_and_tokenizer()
        return self.evaluate()

if __name__ == "__main__":
    evaluator = QAEvaluation()
    evaluator.run()
