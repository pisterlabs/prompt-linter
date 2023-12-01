import openai
import random
import spacy
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import numpy as np
import json
import argparse
detail_keys = {'seen':{
                "pick":[0,1,2,3,10,11,12,13],
                'stack_blocks':[4,5,14,15],
                'put_all_corner':[6,7,16,17],
                'put_all_bowl':[8,9,18,19]},
        'unseen':{
                'put_different_corners':[0,1,10,11],
                'matching':[2,3,8,12,13,18],
                'mismatching':[4,5,9,14,15,19],
                'stack_all_blocks_conrer':[6,7,16,17]},
        'ambiguous':{
                'user_block':[0,1,2,3,4],
                'user_bowl':[5,6,7,8,9],
                'the_block':[10,11,12,13,14],
                'implied':[11,12,14,16,17,19]}
        }
labels = {
    'pick':1,
    'stack_blocks':2,
    'put_all_corner':1,
    'put_all_bowl':1,
    'put_different_corners':1,
    'matching':1,
    'mismatching':1,
    'stack_all_blocks_conrer':1,
    'user_block':2,
    'user_bowl':2,
    'the_block':2
}
label_names = {1:'certain',2:'ambiguous'}

_keys = ["ANY BLOCK", "ANY BOWL", "ANY CORNER", "ANY WHERE"]

def RULES(type, pick, place):
    pick_color = pick.split(" ")[0]
    place_color = place.split(" ")[0]
    # Matched Bowl
    if type == 1:
        if pick_color == place_color: return 1
        else: return 0
    # Mismatched Bowl
    elif type == 2:
        if pick_color != place_color: return 1
        else: return 0
    else:
        raise ValueError("Type should be 1 or 2")

def main(args):
    with open('./res/pick_and_place_unct_vild.json','r') as f:
        data = json.load(f)
    lm_type = args.lm
    unct_type = args.method
    if lm_type == 'chat' or lm_type == 'llama':
        with open('./res/pick_and_place_unct_{}_{}.json'.format(lm_type,unct_type),'r') as f:
            res = json.load(f)
    else:
        with open('./res/pick_and_place_unct_{}.json'.format(unct_type),'r') as f:
            res = json.dump(f)
    for name in res:
        GT = data[name]['gt']
        steps = res[name]['steps']
        gt_objects = res[name]['gt_objects']
        success = 1
        replace_key = {}
        goal = res[name]['name']
        if len(steps) == 0:
            success = 0
        elif GT == 'RULE_1' or GT == 'RULE_2':
            for plan_step in steps:
                flag2 = True
                try:
                    plan_pick, plan_place = plan_step.replace("robot action: robot.pick_and_place(", "").replace(")", "").split(", ")
                except:
                    success = 0
                    flag2 = False
                if flag2:
                    if plan_pick in gt_objects and plan_place in gt_objects:
                        temp= RULES(int(GT.split("_")[1]), plan_pick, plan_place)
                    else:
                        success = 0
                    success *= temp

        else:
            if goal == "stack all the blocks":
                if len(GT) == len(steps) or len(GT) -1 ==len(steps): pass
                else: success = False
            elif "all the blocks" in goal:
                if len(GT) == len(steps): pass
                else: success = False
            for gt_step, plan_step in zip(GT, steps):
                
                plan_pick, plan_place = plan_step.replace("robot action: robot.pick_and_place(", "").replace(")", "").split(", ")
                gt_pick, gt_place = gt_step.replace("robot.pick_and_place(", "").replace(")", "").split(", ")
                # print(gt_pick, plan_pick, gt_place, plan_place)
                if _keys[3] in gt_pick:
                    gt_pick = plan_pick
                if _keys[3] in gt_place:
                    gt_place = plan_place
                    flag = True
                if _keys[0] in gt_pick:
                    flag = False                
                    for rk in replace_key:
                        if gt_pick == rk:
                            gt_pick = replace_key[rk]
                            flag = True
                            break
                    if not flag:
                        replace_key[gt_pick] =  plan_pick
                        gt_pick = plan_pick
                if _keys[1] in gt_place or _keys[2] in gt_place or _keys[0] in gt_place:
                    flag = False
                    for rk in replace_key:
                        if gt_place == rk:
                            gt_place = replace_key[rk]
                            flag = True
                            break
                    if not flag:
                        if 'behind' in gt_place:
                            plan_place2 = 'behind ' + plan_place
                            replace_key[gt_place] =  plan_place2
                            gt_place = plan_place2
                            print(plan_place)
                        elif 'next to' in gt_place:
                            plan_place2 = 'next to ' + plan_place
                            replace_key[gt_place] =  plan_place2
                            gt_place = plan_place2
                        else:
                            replace_key[gt_place] =  plan_place
                            gt_place = plan_place
                # if flag:
                #     print(gt_pick, plan_pick, gt_place, plan_place)
                success *= (gt_pick == plan_pick and gt_place == plan_place)
                # print(replace_key, success)
        res[name]['success'] = success
    if lm_type == 'chat' or lm_type == 'llama':
        with open('./res/pick_and_place_unct_{}_{}.json'.format(lm_type,unct_type),'w') as f:
            json.dump(res, f, indent=4)
    else:
        with open('./res/pick_and_place_unct_{}.json'.format(unct_type),'w') as f:
            json.dump(res, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",default=1, type = int)          # extra value
    parser.add_argument("--lm", default='gpt3', type=str)           # existence/nonexistence
    args = parser.parse_args()
    main(args)