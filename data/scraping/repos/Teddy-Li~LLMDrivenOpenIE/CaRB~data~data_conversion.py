import json
import csv
import argparse
import openai
import os
import sys
import time



def reformat_for_annotation(args):
    for subset in ['dev', 'test']:
        in_path = args.in_path % subset
        out_path = args.cvt_path % subset
        data = []
        with open(in_path, 'r', encoding='utf8') as ifp:
            for line in ifp:
                item_lst = line.rstrip('\n').split('\t')
                sent = item_lst[0]
                pred = item_lst[1]
                arguments = item_lst[2:]
                if len(arguments) > 4:
                    print(f'Error: {arguments}')
                    # assert len(arguments) <= 4, f'Error: {arguments}'
                data.append({
                    'sent': sent,
                    'pred': pred,
                    'subj': arguments[0],
                    'obj0': ('', arguments[1]) if len(arguments) >= 2 else ('', ''),
                    'obj1': ('', arguments[2]) if len(arguments) >= 3 else ('', ''),
                    'obj2': ('', arguments[3]) if len(arguments) >= 4 else ('', ''),
                    'auxilliary': ''
                })
        with open(out_path, 'w', encoding='utf8') as ofp:
            json.dump(data, ofp, indent=2, ensure_ascii=False)


def gpt4_annotate(args):
    in_context_examples = """convert open IE annotation to the appropriate format:
a. annotate prepositions for each object when applicable;
b. do not cut away prepositional modifiers from arguments, only cut away prepositional modifiers from predicates;
c. extract embedded triples in ``said'' / ``claimed'' structures;
d. move modals and negations to "auxilliary";
e. keep the original order of elements;

Sentence: Tom Bradley joined the London , Midland and Scottish Railway Company as a junior clerk in the Goods Depot at Kettering in 1941 .

Before: 
1. joined ;; Tom Bradley ;; the London, Midland and Scottish Railway Company ;; as a junior clerk in the Goods Depot ;; at Kettering ;; in 1941
2. exists ;; a Goods Depot ;; in Ketering ;; in 1941
3. exists ;; the London, Midland and Scottish Railway Company ;; at Kettering ;; in 1941
4. is ;; Tom Bradley ;; a junior clerk in the Goods Depot ;; at Kettering ;; in 1941

After: 
1. joined ;;Tom Bradley ;; the London, Midland and Scottish Railway Company ;; as ### a junior clerk in the Goods Depot at Kettering ;; in ### 1941
2. exists ;; a Goods Depot ;; in ### Kettering ;; in ### 1941
3. exists ;; the London, Midland and Scottish Railway Company ;; in ### Kettering ;; in ### 1941
4. is ;; Tom Bradley ;; a junior clerk ;; in ### the Goods Depot ;; at ### Kettering ;; in ### 1941

convert open IE annotation to the appropriate format:
a. annotate prepositions for each object when applicable;
b. do not cut away prepositional modifiers from arguments, only cut away prepositional modifiers from predicates;
c. extract embedded triples in ``said'' / ``claimed'' structures;
d. move modals and negations to "auxilliary";
e. keep the original order of elements;

Sentence: However , Jesus is not accepted as the son by Muslims , who strictly maintain that he was a human being who was loved by God and exalted by God to ranks of the most righteous .

Before: 
1. is not accepted as ;; Jesus ;; the son
2. strictly maintain that Jesus was  ;; Muslims ;; a human being
3. strictly maintain that Jesus was loved by ;; Muslims ;; God
4. strictly maintain that Jesus was exalted to ranks of the most rightuous by ;; Muslims ;; God

After: 
1. not ### is accepted ;; Jesus ;; by ### Muslims ;; as ### the son
2. strictly maintain that Jesus was ;; Muslims ;; a human being
3. strictly maintain that Jesus was loved ;; Muslims ;; by ### God
4. strictly maintain that Jesus was exalted ;; Muslims ;; to ### ranks of the most rightuous ;; by ### God
5. was ;; Jesus ;; a human being
6. loved ;; God ;; Jesus
7. exalted ;; God ;; Jesus ;; to ### ranks of the most rightuous

convert open IE annotation to the appropriate format:
a. annotate prepositions for each object when applicable;
b. do not cut away prepositional modifiers from arguments, only cut away prepositional modifiers from predicates;
c. extract embedded triples in ``said'' / ``claimed'' structures;
d. move modals and negations to "auxilliary";
e. keep the original order of elements;

Sentence: From 1970 to 1985 , Gideon Rodan taught at the University of Connecticut School of Dental Medicine until he switched over to Merck .

Before: 
1. taught at ;; Gideon Rodan ;; the University of Connecticut School of Dental Medicine ;; from 1970
2. taught at ;; Gideon Rodan ;; the University of Connecticut School of Dental Medicine ;; to 1985
3. taught at ;; Gideon Rodan ;; the University of Connecticut School of Dental Medicine ;; until he switched over to Merck
4. switched over to ;; He ;; Merck
5. has ;; the University of Connecticut ;; a School of Dental Medicine

After: 
1. taught ;; Gideon Rodan ;; at ### the University of Connecticut School of Dental Medicine ;; from ### 1970 ;; to ### 1985 ;; until ### he switched over to Merck
2. switched over ;; Gideon Rodan ;; to ### Merck
3. has ;; the University of Connecticut ;; a School of Dental Medicine

convert open IE annotation to the appropriate format:
a. annotate prepositions for each object when applicable;
b. do not cut away prepositional modifiers from arguments, only cut away prepositional modifiers from predicates;
c. extract embedded triples in ``said'' / ``claimed'' structures;
d. move modals and negations to "auxilliary";
e. keep the original order of elements;

Sentence: """
    subsets = ['dev', 'test'] if args.subset is None else [args.subset]
    for subset in subsets:
        print(f"Processing {subset}")
        in_path = args.in_path % subset
        anno_path = args.anno_path % subset

        existing_annos = {}
        if os.path.exists(anno_path+'_'):
            with open(anno_path+'_', 'r', encoding='utf8') as ifp:
                for line in ifp:
                    item = json.loads(line.rstrip('\n'))
                    existing_annos[item['sent']] = item
                    
        afp = open(anno_path, 'w', encoding='utf8')
        data = {}
        with open(in_path, 'r', encoding='utf8') as ifp:
            for line in ifp:
                if len(line) < 2:
                    continue
                lst = line.rstrip('\n').split('\t')
                sent = lst[0]
                if sent not in data:
                    data[sent] = []
                data[sent].append(lst[1:])
        total_tokens = 0

        start_t = time.time()

        for sidx, sent in enumerate(data):
            if sidx % 10 == 0:
                durr = time.time() - start_t
                print(f"Processing sentence {sidx} / {len(data)}; total tokens so far: {total_tokens}; duration: {durr//60}m {durr%60:.2f}s;")
            
            if sent in existing_annos:
                oline = json.dumps(existing_annos[sent], ensure_ascii=False)
                afp.write(oline + '\n')
                continue

            curr_prompt = f"{sent}\n\nBefore:\n"
            for i, tup in enumerate(data[sent]):
                curr_prompt += f"{i+1}. {' ;; '.join(tup)}\n\nAfter:"
            curr_prompt = in_context_examples + curr_prompt
            gpt_failure_flag = False
            for try_id in range(3):
                try:
                    time.sleep(3)
                    response = openai.ChatCompletion.create(
                        model="gpt-4-0613",
                        messages=[
                            {
                            "role": "user",
                            "content": curr_prompt,
                            }
                        ],
                        temperature=0.7,
                        max_tokens=512,
                        top_p=0.8,
                        frequency_penalty=0,
                        presence_penalty=0
                        )
                    returned_text = response.choices[0]['message']['content']
                    curr_tokens_num = response.usage['total_tokens']
                    total_tokens += curr_tokens_num
                    if len(returned_text) < 10:
                        continue
                    else:
                        break
                except Exception as e:
                    print(f"Error: {e}; retry: {try_id}", file=sys.stderr)
                    if try_id == 2:
                        print(f"Entry: {sent} skipped due to repeated errors", file=sys.stderr)
                    continue
            annotated = {'sidx': sidx, 'sent': sent, 'triples': []}
            if not gpt_failure_flag:
                lines = returned_text.split('\n')
                for tup in lines:
                    if len(tup) < 5 or not tup[0].isdigit():
                        continue
                    tup_dotsplit = tup.split('.')
                    if not tup_dotsplit[0].isdigit():
                        print(f"sidx {sidx}; Error: {tup}", file=sys.stderr)
                    tup = '.'.join(tup_dotsplit[1:])
                    tup = tup.split(' ;; ')
                    pred = tup[0]
                    if '###' in pred:
                        pred_list = pred.split('###')
                        pred_list = [x.strip(' ') for x in pred_list]
                        if len(pred_list) != 2:
                            print(f"sidx {sidx}; Error pred_list: {pred_list}", file=sys.stderr)
                        auxi = pred_list[:-1]
                        pred = pred_list[-1]
                    else:
                        auxi = []
                    subj = tup[1]
                    if '###' in subj:
                        print(f"sidx {sidx}; Error: subject should not have ###: {subj}; {tup}", file=sys.stderr)
                    objs = tup[2:]
                    objs_parsed = []
                    for obj in objs:
                        if '###' in obj:
                            obj_list = obj.split('###')
                            obj_list = [x.strip(' ') for x in obj_list]
                            if len(obj_list) != 2:
                                print(f"sidx {sidx}; Error obj_list: {obj_list}", file=sys.stderr)
                            prep, obj = obj_list[:2]
                        else:
                            prep = ""
                        objs_parsed.append((prep, obj))
                    annotated['triples'].append({
                        'pred': pred,
                        'subj': subj,
                        'objs': objs_parsed,
                        'auxi': auxi
                    })
            else:
                pass
            oline = json.dumps(annotated, ensure_ascii=False)
            afp.write(oline + '\n')
        print(f"GPT annotation completed for {subset}; total tokens: {total_tokens}")
        afp.close()


def prepare_input_outputs_for_llama2(args):
    subsets = ['dev', 'test'] if args.subset is None else [args.subset]
    for subset in subsets:
        in_path = args.anno_path % subset
        out_path = args.supervision_path % subset
        ofp = open(out_path, 'w', encoding='utf8')
        with open(in_path, 'r', encoding='utf8') as ifp:
            for line in ifp:
                item = json.loads(line.rstrip('\n'))
                sent = item['sent']
                triples = item['triples']
                output = []
                for triple in triples:
                    if triple['auxi'] == '':
                        ostr = f"{triple['pred']} ;; {triple['subj']}"
                    else:
                        ostr = f"{triple['auxi']} ### {triple['pred']} ;; {triple['subj']}"
                    for obj in triple['objs']:
                        if obj[0] == '':
                            ostr += f" ;; {obj[1]}"
                        else:
                            ostr += f" ;; {obj[0]} ### {obj[1]}"
                    output.append(ostr)
                out_aggr_str = '\n'.join([f"{i+1}. {ostr}" for i, ostr in enumerate(output)])
                io_pair = {'sidx': item['sidx'], 'sent': sent, 'output': output}
                oline = json.dumps(io_pair, ensure_ascii=False)
                ofp.write(oline + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='./gold/%s.tsv')
    parser.add_argument('--cvt_path', type=str, default='./CaRBent_gold/%s.json')
    parser.add_argument('--anno_path', type=str, default='./CaRBent_gold/%s_reanno.json')
    # parser.add_argument('--supervision_path', type=str, default='./CaRBent_gold/%s_spvsn.json')
    parser.add_argument('--task', type=str, default='rfmt_anno')
    parser.add_argument('--subset', type=str, default=None, choices=['dev', 'test'], help='if None, process both dev and test')
    args = parser.parse_args()
    if args.task == 'rfmt_anno':
        reformat_for_annotation(args)
    elif args.task == 'gpt_annotate':
        gpt4_annotate(args)
    # elif args.task == 'prepare_input_outputs':
    #     prepare_input_outputs_for_llama2(args)
    else:
        raise NotImplementedError
