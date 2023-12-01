# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:09:01 2019

@author: HareeshRavi
"""
import configAll
import cnsi
import baseline
import process_vist
import vggfeat_vist
import coherenceVec
import argparse
import time
import json

if __name__ == '__main__':
    
    #argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', type=str, default=None, 
                        help='use this argument to preprocess VIST data')
    parser.add_argument('--pretrain', action='store_true', default=False, 
                        help='use this to pre-trainstage 1 of the network')
    parser.add_argument('--train', type=str, default=None, 
                        help ='train stage1, cnsi, nsi or baseline')
    parser.add_argument('--eval', type=str, default=None, 
                        help ='evaluate cnsi, nsi or baseline')
    parser.add_argument('--show', type=str, default=None, 
                        help ='show the story for cnsi, nsi or baseline')
    args = parser.parse_args()
    
    # get config
    try:
        configs = json.load(open('config.json'))
    except FileNotFoundError:
        configs = configAll.create_config()
    '''
    To preprocess
    '''
    if args.preprocess == 'data':
        
        # process vist data jsons and put it according to usage
        starttime = time.time()
        process_vist.main(configs)
        print('vist data files created in {} secs'.format(time.time() - 
              starttime))
    elif args.preprocess == 'imagefeatures':
        # extract vgg feats for all images. also remove images (and stories)
        # for where images are not present
        starttime = time.time()
        vggfeat_vist.main(configs)
        print('vggfeat extracted for all images in {} secs'.format(
                time.time() - starttime))
    elif args.preprocess == 'coherencevectors':
        # get coherence vector for all stories
        # run following command in terminal before running below code
        # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
        starttime = time.time()
        coherenceVec.main(configs)
        print('coherence vector extracted for all stories in {} secs'.format(
                time.time() - starttime))
    elif not args.preprocess:
        pass
    else:
        raise ValueError('preprocess types are data, imagefeatures and ' + 
                         'coherencevectors')
    
    if args.pretrain:
        '''
        Pretrain stage 1 on MSCOCO dataset
        '''
        cnsi.main(configs, 'pretrain')
    else:
        pass
        
    if args.train == 'stage1':
        '''
        train stage 1 on VIST dataset
        '''
        cnsi.main(configs, 'trainstage1')
        
    elif args.train == 'cnsi':
        '''
        To Train CNSI model stage 2
        '''
        cnsi.main(configs, 'trainstage2', 'cnsi')
        
    elif args.train == 'nsi':
        '''
        To Train NSI model stage 2
        '''
        cnsi.main(configs, 'trainstage2', 'nsi')
        
    elif args.train == 'baseline':
        '''
        To Train baseline model
        '''
        configs['model'] = 'baseline'
        baseline.main(configs, 'train')
    elif not args.train:
        pass
    else:
        raise ValueError('args for train can be stage1, cnsi, ' + 
                         'nsi or baseline only')
    
    '''
    To evaluate 'model' on VIST test set. This will save predictions in file
    for further use by metrics. Will not print or show any results.
    '''
    if args.eval == 'cnsi':
        # get predictions for stories from testsamples for cnsi model
        model2test =  (configs['savemodel'] + 'stage2_cnsi_' + 
                       configs['date'] + '.h5')
        cnsi.main(configs, 'test', 'cnsi', model2test)
    elif args.eval == 'nsi':
        # get predictions for stories from testsamples for nsi model
        model2test =  (configs['savemodel'] + 'stage2_nsi_' + 
                       configs['date'] + '.h5')
        cnsi.main(configs, 'test', 'nsi', model2test)
    elif args.eval == 'baseline':
        # get predictions for stories from testsamples for baseline model
        baseline.main(configs, 'test')
    elif not args.eval:
        pass
    else:
        raise ValueError('args for eval can be cnsi, ' + 
                         'nsi or baseline only')
    
    
    '''
    To evaluate 'model' on VIST test set. This will save predictions in file
    for further use by metrics. Will not print or show any results.
    '''
    if args.show == 'cnsi':
        # get predictions for stories from testsamples for cnsi model
        results2show =  ('results_cnsi_' + configs['date'] + '.pickle')
        cnsi.main(configs, 'show', 'cnsi', results2show)
        
    elif args.show == 'nsi':
        # get predictions for stories from testsamples for nsi model
        model2test =  ('results_nsi_' + configs['date'] + '.pickle')
        cnsi.main(configs, 'show', 'nsi', model2test)
        
    elif args.show == 'baseline':
        # get predictions for stories from testsamples for baseline model
        baseline.main(configs, 'show')
        
    elif not args.show:
        pass
    else:
        raise ValueError('args for eval can be cnsi, ' + 
                         'nsi or baseline only')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    