# -*- coding: utf-8 -*-
"""
Created on Fri Nov 3 09:40:05 2017

@author: HareeshRavi
"""

from pycorenlp import StanfordCoreNLP
import utils_vist
from coherence_vector import entity_score
import numpy as np
nlp = StanfordCoreNLP('http://localhost:9000')

def get_parse_tree(data):
    
    # extract parse trees using standfordcoreNLP
    tree = []
    for storyidx in range(len(data)):
        tree_temp = ''
        curstory = data[storyidx]
        for sentidx in range(len(curstory)):
            output = nlp.annotate(curstory[sentidx], 
                                  properties={'annotators': 'parse', 
                                              'outputFormat': 'json'}) 
            if sentidx < len(curstory)-1:
                 tree_temp += output['sentences'][0][
                         'parse'].replace('\n', '') + '\n'
            else:
                 tree_temp += output['sentences'][0][
                         'parse'].replace('\n', '    ')

        tree.append(tree_temp)
        print('stories processed: {}/{}'.format(storyidx, len(data)), end='\r')  
           
    return tree

def main(config):
    
    datadir = './data/'
    process = ['train', 'val', 'test']
    
    for proc in process:
        # read the stories
        stories = utils_vist.getSent(datadir + proc + '/' + 
                                     proc + '_text.csv')
        parsetree = get_parse_tree(stories)
        
        print('obtained parse trees..')
        # save the trees in file
        np.save(datadir + proc + '/' + proc + '_parsetree.npy', parsetree)
        
        # get entitiy feature for all stories in the file
        entity_feat = entity_score.entity_feature(parsetree)
        print('obtained entity features..')
        
        # convert dict to numpy array
        cohvec = np.zeros((len(stories), 64), dtype=float)
        for i in entity_feat:
            cohvec[int(i)] = entity_feat[i]
        
        # save entity feature as numpy file
        np.save(datadir + proc + '/cohvec_' + proc + '.npy', 
                cohvec)
    
if __name__ == '__main__':
    
    main()
