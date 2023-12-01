from typing import Tuple, Sequence, Mapping
from pprint import pprint
from pathlib import Path
import re # for loading is_spam

from fire import Fire
import openai
import pandas as pd 
from omegaconf import OmegaConf
from tqdm import tqdm
from time import sleep
from utils import *
import numpy as np



def do_code_gpt(prompt:str, model:str='gpt-4')->str:
    response = openai.ChatCompletion.create(
            model = model,
            messages = [
                dict(role='system', content='You are a helpful assistant.'),
                dict(role='user', content=prompt)
                ]
        )
    decision_code = response['choices'][0]['message']['content']
    return decision_code

def prepare_sampler(msgs:Sequence[str]):
    # permute the training examples
    # if lengthy msg exists, adapt it to THLD. 

    length_adapter = lambda x: get_exerpt(x, toklength=THLD*2) if count_token(x)>THLD else x
    
    msgs_ = msgs.apply(length_adapter) # length adapting 
    msgs_perm = msgs.sample(frac=1., random_state=SEED) # sample whole --> permuting order of the series 
    # print(SEED)
    # pprint(msgs_perm[:4])
    return msgs_perm    

def make_prompts(spam_examples:Sequence[str],
                normal_examples:Sequence[str])->Sequence[str]:
    
    prompts = [] 
    num_hops = CONF.num_hops 

    for i in range(num_hops+1):
        if i == 0:
            templ_f = 'prompting_test/coldstart.yaml'
        else:
            templ_f = 'prompting_test/warmstart.yaml'

        templ = OmegaConf.load(templ_f)
        
        def refine_txt(txt):
            txt =txt.strip('\n')
            txt =txt+"\n"
            return txt
        spam_examples = [refine_txt(l) for l in spam_examples]
        normal_examples = [refine_txt(l) for l in normal_examples]

        prompt= f'''{templ.spamheader}
* {"* ".join(spam_examples)}



{templ.normalheader}
* {"* ".join(normal_examples)}



{templ.suffix}
'''
        prompts.append(prompt)

    return prompts


    
def main(
    config_file:str= 'config_yamls/cold2_15.yaml',
):
    openai.api_key = open('apikey.txt').readlines()[0].strip()
    global CONF, THLD, SEED
    CONF = OmegaConf.load(config_file)
    THLD = CONF.example_len_threshold # --> used for whether to get exerpt or just use whole example
    SEED = CONF.random_seed # --> used for sampler 
    dbg = CONF.dbg
    pprint(CONF)

    # load data
    df = pd.read_csv(CONF.dataset)
    spams = df.spam[~(df.spam.isna())]# protect from nan
    normals = df.normal[~(df.normal.isna())]
    

    if CONF.coverage>1 and isinstance(CONF.coverage, int):
        for ep in range(CONF.start_ep, CONF.coverage):
            SEED += ep*100 # ep=0,1,2,3,...coverage-1
            # prepare sampler
            num_total= int(CONF.coverage * len(spams))
            spams_ = prepare_sampler(spams)[:num_total]
            normals_ = prepare_sampler(normals)[:num_total]
            spams__, normals__  = map(lambda x: make_minibatches(x, bsz=CONF.n_context_examples), [spams_, normals_])
            
            # make sure two batches have the same number of minibatches 
            ldiff = len(spams__) - len(normals__)
            for _ in range(abs(ldiff)):
                if len(spams__)> len(normals__):
                    spams__.pop()
                elif len(normals__) > len(spams__):
                    normals__.pop()
                else:
                    pass
            
            # prepare prompts for decision function generation 
            functions = []
            prompts_per_f= [] 
            for sp_exs, n_exs in tqdm(zip(spams__, normals__), total=len(spams__), desc='generating decision functions'):
                prompts = make_prompts(sp_exs, n_exs)
                f_use = []
                for p in prompts:
                    count_fail = 0
                    while 1:
                        try:
                            f:str = do_code_gpt(p, model=CONF.model) if not dbg else 'def is_spam(txt): return True'
                            break
                        except:
                            count_fail+=1
                            print(f'api failed {count_fail} times, wait for several seconds to retry.')
                            if count_fail<4:
                                sleep(np.random.randint(5))
                            else: # count fail>=4
                                print("failing too much")
                                x = np.random.randint(4)
                                sleep(60*x + 30)
                                print(f"waiting for {x}.5 mins")
                                count_fail=0
                    f_use.append(f)
                    f_use = f_use[-CONF.keep_k_solutions:]
                # record
                prompts_per_f.extend(prompts[-CONF.keep_k_solutions:])
                functions.extend(f_use)
            
            # store the functions made
            storage:Path = Path()/'expresults'/CONF.expname/f"{CONF.model if not dbg else 'dbg'}_nhop{CONF.num_hops}_n_ctx{CONF.n_context_examples}_{ep}_{CONF.coverage}x{CONF.dataset.split('/')[-1].replace('.csv','')}"
            if not storage.exists():
                storage.mkdir(parents=True)
                with open(storage/'saved_conf.yaml', 'w') as ymlf:
                    ymlf.write(OmegaConf.to_yaml(CONF))
            for code, pmpt, i in zip(functions, prompts_per_f, range(len(functions))):
                code_ = refine_code(code) # to make gpt3.5 work, refine_code need to be more sophisticated, but decided not to code for that
                try: # for 3.5 api being too buggy 
                    exec(code_, globals()) # is_spam 함수가 선언됨, 항상 덮어써짐
                    tp, tn, tpmsg, fpmsg = eval_on_unseen_train(func=is_spam, spams=spams, normals=normals, n_context_examples = CONF.n_context_examples)
                    pmpt_ = f"performance: TP = {tpmsg} | FP = {fpmsg}\n\n{pmpt}"
                    fname = f"function_{i}_{tp}_{tn}.py"
                    fname_ = f"prompt_{i}.md"
                    with open(storage/fname, 'w') as cf, open(storage/fname_, 'w') as pf:
                        cf.write(code_)
                        pf.write(pmpt_)
                except:
                    continue
    else:
        # prepare sampler
        num_total= int(CONF.coverage * len(spams))
        spams_ = prepare_sampler(spams)[:num_total]
        normals_ = prepare_sampler(normals)[:num_total]
        spams__, normals__  = map(lambda x: make_minibatches(x, bsz=CONF.n_context_examples), [spams_, normals_])
        
        # make sure two batches have the same number of minibatches 
        ldiff = len(spams__) - len(normals__)
        for _ in range(abs(ldiff)):
            if len(spams__)> len(normals__):
                spams__.pop()
            elif len(normals__) > len(spams__):
                normals__.pop()
            else:
                pass
        
        # prepare prompts for decision function generation 
        functions = []
        prompts_per_f= [] 
        for sp_exs, n_exs in tqdm(zip(spams__, normals__), total=len(spams__), desc='generating decision functions'):
            prompts = make_prompts(sp_exs, n_exs)
            f_use = []
            for p in prompts:
                f:str = do_code_gpt(p, model=CONF.model) if not dbg else 'def is_spam(txt): return True'
                f_use.append(f)
                f_use = f_use[-CONF.keep_k_solutions:]
            # record
            prompts_per_f.extend(prompts[-CONF.keep_k_solutions:])
            functions.extend(f_use)
        
        # store the functions made
        storage:Path = Path()/'expresults'/CONF.expname/f"{CONF.model if not dbg else 'dbg'}_nhop{CONF.num_hops}_n_ctx{CONF.n_context_examples}_0_{CONF.coverage}x{CONF.dataset.split('/')[-1].replace('.csv','')}"
        if not storage.exists():
            storage.mkdir(parents=True)
            with open(storage/'saved_conf.yaml', 'w') as ymlf:
                ymlf.write(OmegaConf.to_yaml(CONF))
        for code, pmpt, i in zip(functions, prompts_per_f, range(len(functions))):
            code_ = refine_code(code) # to make gpt3.5 work, refine_code need to be more sophisticated, but decided not to code for that
            try: # for 3.5 api being too buggy 
                exec(code_, globals()) # is_spam 함수가 선언됨, 항상 덮어써짐
                tp, tn, tpmsg, fpmsg = eval_on_unseen_train(func=is_spam, spams=spams, normals=normals, n_context_examples = CONF.n_context_examples)
                pmpt_ = f"performance: TP = {tpmsg} | FP = {fpmsg}\n\n{pmpt}"
                fname = f"function_{i}_{tp}_{tn}.py"
                fname_ = f"prompt_{i}.md"
                with open(storage/fname, 'w') as cf, open(storage/fname_, 'w') as pf:
                    cf.write(code_)
                    pf.write(pmpt_)
            except:
                continue

        



if __name__ == '__main__':
    Fire(main)