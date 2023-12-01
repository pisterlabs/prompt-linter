from src.ctmodel import CTModel
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from contextualized_topic_models.evaluation.measures import CoherenceNPMI, InvertedRBO
from src.evaluation import CoherenceWordEmbeddings
import nltk
from nltk.corpus import stopwords as stop_words
import pandas as pd
import numpy as np
from pathlib import Path
import json
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eucomm-only', type=int, default=1)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # GPU nr, adapt for other envs


def _save_results(rlist, eucomm_only):
    DF_PATH = Path('logs') / 'topic'
    fname = DF_PATH / 'performances.jsonl'
    try:
        rdf = pd.read_json(str(fname), 
                           orient="records",
                           lines=True)
        rdf = pd.concat([rdf, 
                         pd.DataFrame(rlist)], ignore_index=True)
    except:
        rdf = pd.DataFrame(rlist)
    rdf.to_json(str(fname), orient="records", lines=True)

    
def _compute_metrics(model_id, run, split, ds, tlists, tlists_20, 
                     score_list, ctm, entity='all'):
    scores = {}
    scores[f'name'] = model_id
    scores['run'] = run
    scores['split'] = split
    for n, tl in zip([10, 20], [tlists, tlists_20]):
        npmi = CoherenceNPMI(texts=[d.split() for d in ds], 
                             topics=tl)
        rbo = InvertedRBO(topics=tl)
        cwe = CoherenceWordEmbeddings(topics=tl) 
        scores['entity'] = entity
        scores[f'npmi_{n}'] = npmi.score(topk=n).round(4)
        if split == 'train':
            scores[f'cwe_{n}'] = cwe.score(topk=n).round(4)
            scores[f'rbo_{n}'] = rbo.score(topk=n).round(4)
        else:
            scores[f'cwe_{n}'] = np.nan
            scores[f'rbo_{n}'] = np.nan
    score_list.append(scores)
    

def main(eucomm_only=False):
     
    sent_transformers = glob.glob('models/sent_transformers/finetuned/distilbert*')
    sent_transformers += glob.glob('models/sent_transformers/pretrained/distilbert*')
    
    fs = glob.glob('data/derivatives/*')
    dfs = []
    for f in fs:
        df = pd.read_json(f, orient='records', lines=True)
        df['entity'] = f.split('/')[-1][:-16]
        dfs.append(df)
    topic_df = pd.concat(dfs)
    
    # Get training indices
    train_idx = set(np.where(topic_df['topic_split']=='train')[0].tolist())
    val_idx = set(np.where(topic_df['topic_split']=='val')[0].tolist())
    test_idx = set(np.where(topic_df['topic_split']=='test')[0].tolist())
    eucomm_train_idx = set(np.where((topic_df['topic_split']=='train') &
                                    (topic_df['entity']=='EU_Commission'))[0].tolist())
    eucomm_val_idx = set(np.where((topic_df['topic_split']=='val') &
                                    (topic_df['entity']=='EU_Commission'))[0].tolist())
    eucomm_test_idx = set(np.where((topic_df['topic_split']=='test') &
                                    (topic_df['entity']=='EU_Commission'))[0].tolist())

    # Preprocess
    nltk.download('stopwords')
    stopwords = list(stop_words.words("english"))
    documents = topic_df.text.tolist() 
    topic_df = topic_df.reset_index()
    indices = topic_df.index.tolist()  
    
    logpath = Path('logs') / 'topic'
    modelpath = Path('models') / 'topic'
        
    vocabulary_sizes = [500] # 250
    score_list = []
    
    # Parameters
    for vs in vocabulary_sizes:
        sp = WhiteSpacePreprocessingStopwords(documents, 
                                              stopwords_list=stopwords, 
                                              vocabulary_size=vs)
        prepped, unprepped, vocab, retained_idx = sp.preprocess()
        
        train_indices = [indices[r_idx] for r_idx in retained_idx
                         if r_idx in train_idx]
        val_indices = [indices[r_idx] for r_idx in retained_idx
                       if r_idx in val_idx]
        test_indices = [indices[r_idx] for r_idx in retained_idx
                        if r_idx in test_idx]

        prepped_all_train = [prepped[i] for i in range(len(prepped)) 
                             if retained_idx[i] in train_idx]
        prepped_all_val = [prepped[i] for i in range(len(prepped)) 
                           if retained_idx[i] in val_idx]
        prepped_all_test = [prepped[i] for i in range(len(prepped)) 
                            if retained_idx[i] in test_idx]
        
        unprepped_all_train = [unprepped[i] for i in range(len(unprepped)) 
                               if retained_idx[i] in train_idx]
        unprepped_all_val = [unprepped[i] for i in range(len(unprepped)) 
                             if retained_idx[i] in val_idx]
        unprepped_all_test = [unprepped[i] for i in range(len(unprepped)) 
                              if retained_idx[i] in test_idx]
        
        unprepped_eucomm_train = [unprepped[i] for i in range(len(unprepped)) 
                                if retained_idx[i] in eucomm_train_idx]
        unprepped_eucomm_val = [unprepped[i] for i in range(len(unprepped)) 
                                if retained_idx[i] in eucomm_val_idx]
        unprepped_eucomm_test = [unprepped[i] for i in range(len(unprepped)) 
                                if retained_idx[i] in eucomm_test_idx]
        
        prepped_eucomm_train = [prepped[i] for i in range(len(prepped)) 
                                if retained_idx[i] in eucomm_train_idx]
        prepped_eucomm_val = [prepped[i] for i in range(len(prepped)) 
                                if retained_idx[i] in eucomm_val_idx]
        prepped_eucomm_test = [prepped[i] for i in range(len(prepped)) 
                                if retained_idx[i] in eucomm_test_idx]
        
        
        # Set parameters and prepare
        models = sent_transformers
        n_comps = [20] # 10, 30
        ctx_size = 768 
        batch_sizes = [64]
        lrs = [2e-2]
        
        for model in models:
            mtraining = model.split('/')[-2]
            for run in range(5):
                print(model)
                for n_components in n_comps:
                        for batch_size in batch_sizes:
                            for lr in lrs:

                                # Preparation
                                tp = TopicModelDataPreparation(model)
                                train_eucomm_dataset = tp.fit(unprepped_eucomm_train, 
                                                              prepped_eucomm_train)
                                train_dataset = tp.transform(unprepped_all_train, 
                                                             prepped_all_train)
                                val_dataset = tp.transform(unprepped_all_val, 
                                                           prepped_all_val)
                                test_dataset = tp.transform(unprepped_all_test, 
                                                            prepped_all_test)
                                val_eucomm_dataset = tp.transform(unprepped_eucomm_val, 
                                                                  prepped_eucomm_val)
                                test_eucomm_dataset = tp.transform(unprepped_eucomm_test, 
                                                                   prepped_eucomm_test)

                                # Fit and predict
                                ctm = CTModel(model=model,
                                              bow_size=len(tp.vocab), 
                                              contextual_size=ctx_size, 
                                              n_components=n_components, 
                                              num_epochs=100,
                                              batch_size=batch_size,
                                              lr=lr,
                                              activation='softplus',
                                              vocabulary_size=vs,
                                              num_data_loader_workers=5)
                                
                                ctm.fit(train_eucomm_dataset, 
                                        validation_dataset=val_eucomm_dataset)
                                    
                                pred_train_topics = ctm.get_thetas(train_dataset,
                                                                   n_samples=20)
                                pred_val_topics = ctm.get_thetas(val_dataset,
                                                                 n_samples=20)
                                pred_test_topics = ctm.get_thetas(test_dataset,
                                                                  n_samples=20)

                                # Save topics
                                model_id = f'{ctm.model_name}_train-{mtraining}'
                                LOG_PATH = logpath / model_id / f'run-{run}'
                                LOG_PATH.mkdir(parents=True, exist_ok=True)
                                MODEL_PATH = modelpath / model_id / f'run-{run}'
                                MODEL_PATH.mkdir(parents=True, exist_ok=True)
                                topic_out = MODEL_PATH / 'model.json'
                                with open(str(topic_out), "w") as fh:
                                    json.dump(ctm.get_topics(k=20), fh)

                                # Merge predicted topics with tweets table
                                data = zip(unprepped_all_train + unprepped_all_val + unprepped_all_test,
                                           train_indices + val_indices + test_indices)
                                cs = ['text', 'index']
                                texts = pd.DataFrame(data, columns=cs)
                                pred_mat = np.vstack([pred_train_topics,
                                                      pred_val_topics,
                                                      pred_test_topics]).round(4)
                                col_names = [f'topic_{i}' 
                                             for i in range(n_components)]
                                preds = pd.DataFrame(pred_mat,
                                                     columns=col_names)
                                preds = pd.concat([texts, preds], axis=1)
                                to_be_merged = topic_df.iloc[retained_idx, :].drop('index',
                                                                                   axis=1).reset_index()
                                assert preds.shape[0] == to_be_merged.shape[0]
                                merged = to_be_merged.merge(preds, on=['text', 'index'])
                                merged.to_json(str(LOG_PATH / 'preds.jsonl'),
                                               orient='records', lines=True)

                                # Evaluate model
                                tlists = ctm.get_topic_lists(10)
                                tlists_20 = ctm.get_topic_lists(20)
                                _compute_metrics(model_id,
                                                 run,
                                                 'train', 
                                                 prepped_all_train, 
                                                 tlists, 
                                                 tlists_20,
                                                 score_list,
                                                 ctm)
                                _compute_metrics(model_id,
                                                 run,
                                                 'val', 
                                                 prepped_all_val, 
                                                 tlists, 
                                                 tlists_20,
                                                 score_list,
                                                 ctm)
                                _compute_metrics(model_id,
                                                 run,
                                                 'test', 
                                                 prepped_all_test, 
                                                 tlists, 
                                                 tlists_20,
                                                 score_list,
                                                 ctm) 
                                
                                for i in ['EU_Commission']:
                                    _compute_metrics(model_id,
                                                     run,
                                                     'train', 
                                                     prepped_eucomm_train, 
                                                     tlists, 
                                                     tlists_20,
                                                     score_list,
                                                     ctm,
                                                     entity=i)
                                    _compute_metrics(model_id,
                                                     run,
                                                     'val', 
                                                     prepped_eucomm_val, 
                                                     tlists, 
                                                     tlists_20,
                                                     score_list,
                                                     ctm,
                                                     entity=i)
                                    _compute_metrics(model_id,
                                                     run,
                                                     'test', 
                                                     prepped_eucomm_test, 
                                                     tlists, 
                                                     tlists_20,
                                                     score_list,
                                                     ctm,
                                                     entity=i)

                                # Save model
                                ctm.save(models_dir=str(MODEL_PATH), final=True)
    _save_results(score_list, eucomm_only)

    
if __name__=="__main__":
    args = parser.parse_args()
    main(bool(args.eucomm_only))
