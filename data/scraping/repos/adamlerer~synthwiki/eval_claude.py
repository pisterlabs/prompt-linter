import numpy as np
import pandas as pd
import openai
import argparse
import random
import hashlib 
import time
import os
from synthwiki_utils import checkCorrectness, insertIntoJunk, genJunkContext, hash_string, check_file_exists
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import transformers

parser = argparse.ArgumentParser(description="A simple argparse example.")
parser.add_argument('--input_file', 
                    default=f'{__file__}/data/madlibs/madlibs1.csv',
                    help='Where questions?')
parser.add_argument('--result_dir', 
                    help='Where results to save?', required=True)
parser.add_argument("--anthropic_key", help="UR KEY BOR")
parser.add_argument("--junk_size", default=30000, type=int, help="How much junk")
parser.add_argument('--no_junk', default=0, type=int, help="Clean run with just the single doc")
parser.add_argument("--insert_place", default='random', help="Where insert")
parser.add_argument("--sample_n", default=100, type=int)
parser.add_argument("--model", default='claude-2', type=str)
args = parser.parse_args()

tokenizer = transformers.AutoTokenizer.from_pretrained('togethercomputer/LLaMA-2-7B-32K')

os.makedirs(args.result_dir, exist_ok=True)

anthropic = Anthropic(api_key=args.anthropic_key)

raw = pd.read_csv(args.input_file)
all_contexts = np.unique(raw['context'].values)

## ADD A SAMPEL
re_ordered = raw.sample(args.sample_n)
real_context = re_ordered['context'].values
real_question = re_ordered['question'].values
real_answer = re_ordered['answer'].values

example_junk = genJunkContext(real_context.tolist(), limit=2500, tokenizer=tokenizer)
print(example_junk)


def askClaude(question, supporting_docs, model):
    doc_string = ""
    
    doc_ids = []

    for d in supporting_docs:
        doc_id = hash_string(d, len=8)
        doc_ids.append(doc_id)
        doc_string += f"Document [{doc_id}]: {d} \n\n"

    prompt = f"""Here is some information you will use to answer a question. Some of the information may be irrelevant.\n\##DOCUMENTS{doc_string}##QUESTION\n{question}\n\nPlease return only the answer to the question. Answer concisely.
    """

    retry_limit = 10
    retry_count = 0
    while retry_count < retry_limit:
        try:
            completion = anthropic.completions.create(
                model=model,
                max_tokens_to_sample=300,
                prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
            )
            break
        except Exception as e:
            print(f"Error occurred: {e}. Retrying...")
            retry_count += 1
            time.sleep(60)

    if retry_count == retry_limit:
        answer = "Reached maximum retry limit."
        return 'something failed'
    
    answer = completion.completion

    return answer

if __name__ == "__main__":
    ## Not parallelizing this since you hit GPT rate limit basically instantly, one at a time is fine
    for (question, context, answer) in zip(real_question, real_context, real_answer):
        qhash = hash_string(question)
        fname = f'{args.result_dir}/{qhash}.csv'
        
        if check_file_exists(fname):
            print("lol this exists")
        else:
            print("time2go")
            junk_contexts = [c for c in all_contexts if c != context]

            context_to_use = genJunkContext(
                junk_contexts, 
                limit=args.junk_size, 
                tokenizer=tokenizer,
            )
            
            random.shuffle(context_to_use)
            if args.no_junk:
                supp_docs = [context]
                pos_to_insert = 0
            supp_docs, pos_to_insert = insertIntoJunk(context_to_use, context, args.insert_place)

            model_answer = askClaude(question, context_to_use, args.model)

            print(f"Question: {question} | Answer: {answer}")
            print(f"Model answer: {model_answer}")

            correct = checkCorrectness(model_answer, answer)
            row = pd.DataFrame({
                'question': question,
                'junk_size': args.junk_size,
                'claude_correct': correct,
                'model_answer': model_answer,
                'model': args.model,
                'doc_position': pos_to_insert
            }, index=[0])

            row.to_csv(fname, index=False)
