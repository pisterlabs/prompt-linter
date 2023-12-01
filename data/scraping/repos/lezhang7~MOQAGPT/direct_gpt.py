import os
import openai
import sys
sys.path.insert(0, '..')
from utils.utils import *
from dataset_zoo import *
from tqdm import tqdm


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




if __name__ == "__main__":
    dataset=get_dataset("mmcoqa")
    openai.organization = "org-9te1ZAEuaMNaZGBhynbX1k4W"
    openai.api_key =  os.getenv("OPENAI_API_KEY")
    direct_gpt_cot("/home/mila/l/le.zhang/scratch/MOQA/output/mmcoqa/direct_gpt4_cot.json","gpt-4")