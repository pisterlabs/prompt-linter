import os
import openai
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from utils.utils import *
from dataset_zoo import get_dataset
from model_zoo import get_answer_model
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse

def direct_gpt_cot(path,model):
    if os.path.exists(path):
        gpt_results=load_json(path)
    else:
        gpt_results={}
    trial=0
    while len(gpt_results)!=len(dataset):
        try:
            for id,example in tqdm(enumerate(dataset),total=len(dataset)):
                qid=example['qid']
                if qid not in gpt_results:
                    question=example['golden_question_text']
                    # answer=example['answer_text']
                    prompt1=f"{question}\n\n Let's think step by step."
                    
                    completion = openai.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt1}
                        ]
                        )
                    intermediate_reasoning=completion.choices[0].message["content"]
                    prompt2 = f"{intermediate_reasoning}\n\n'{question}\n\nGive me a very short answer, in one or two words."
                    #remove punction
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": prompt2}
                        ]
                        )
                    answer=completion.choices[0].message["content"]
                    gpt_results[qid]={"question":question,"answer":example['answer_text'],"intermediate_reasoning":intermediate_reasoning,"gpt_answer":answer}
                    read_then_save_json(path,gpt_results)
                else: 
                    print(f"{qid} already answered",flush=True) 
        except Exception as e:
            trial+=1
            print(f"{e}, try again, #{trial}")
def collate_fn(batch):
    return {'qid':[example['qid'] for example in batch],'golden_question_text':[example['golden_question_text'] for example in batch],'answer_text':[example['answer_text'] for example in batch]}
def direct_llama(args):
    path=f"/home/mila/l/le.zhang/scratch/MOQA/output/{args.dataset}/direct_{args.direct_qa_model}.json"
    model=get_answer_model(args.direct_qa_model)
    dataset=get_dataset(args.dataset)
   
    dataloader=DataLoader(dataset,batch_size=8,shuffle=False,collate_fn=collate_fn)
    if os.path.exists(path):
        gpt_results=load_json(path)
    else:
        gpt_results={}
    for n,batch in enumerate(tqdm(dataloader)):
        qids=batch['qid']
        questions=batch['golden_question_text']
        gold_answers=batch['answer_text']
        answer,_=model.get_answer_batched(questions,None,with_reasoning=False,direct_answer=True)
        if n<1:
            print("="*20,"output examples","="*20)
            print(f"qids:{qids}\n")
            print(f"questions:{questions}\n")
            print(f"gold_answers:{gold_answers}\n")
            print(f"answer:{answer}\n")
        for id in range(len(qids)):
            gpt_results[qids[id]]={"questions":questions[id],"answer":gold_answers[id],"generated_answer":answer[id]}
        read_then_save_json(path,gpt_results)
    
              

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name",
        choices=["mmqa","mmcoqa"]
    )
    parser.add_argument(
        "--direct_qa_model",
        type=str,
        help="Direct QA model name",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args=parse_args()
    if 'gpt' in args.direct_qa_model:
        direct_gpt_cot(f"/home/mila/l/le.zhang/scratch/MOQA/output/{args.dataset}/direct_{args.direct_qa_model}.json",args.direct_qa_model)
    else:
        direct_llama(args)
   