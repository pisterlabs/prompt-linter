#import entity_grid
from coherence_vector import entity_grid
from coherence_vector import parsetree
import numpy as np


args={}
args['jobs']=2
args['mem']=5
args['parser']="./stanford-parser/stanford-parser.jar"
args['models']="./stanford-parser/stanford-parser-3.9.2-models.jar"
args['grammar']="./stanford-parser/edu/stanford/nlp/models/lexparser/englishPCFG.caseless.ser.gz"
args['threads']=2
args['max_length']=1000
testgrid_path="/media/data2/hareesh_research/cnsi/Show-Me-A-Story/browncoherence/bin64/TestGrid"

def entity_feature(trees_list):

    # input content_list ["This is cspark code paragraph1", "2"...]
    # output sorted (key,score) pair list [(key,score), ..]
    global args
    global testgrid_path
    uniq_list=range(0,len(trees_list)) #for multiple process
    trees_key_list=[]
    for trees, key in zip(trees_list, uniq_list):
        if len(trees.strip())!=0:
            trees_key_list.append({'trees':trees,'key':str(key)})
    grids = parsetree.get_grids_multi_documents(testgrid_path,trees_key_list,
                                                args['jobs'])
    print('obtained grids..')
    feature_vec_list={}
    tot = len(trees_key_list)
    k = 0
    for grid, trees_and_key  in zip(grids, trees_key_list):
        if grid and grid.strip()!="":
            model = entity_grid.new_entity_grid(grid, syntax=True, 
                                                max_salience=0, history=3)
            key=int(trees_and_key['key'])
            feature_vec_list[key]=model.get_trans_prob_vctr()
        else:
            key=int(trees_and_key['key'])
            feature_vec_list[key]=np.zeros(64)
        k += 1
        print('extracting coherence vector: {}/{}'.format(k, tot), end='\r')
    for key in uniq_list:
        if key in feature_vec_list:
            pass
        else:
            feature_vec_list[key]=np.zeros(64)
    return feature_vec_list
