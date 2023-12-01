import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib
from helpers.analysis.classification_report import calculate_precision_recall
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from gensim import corpora, models, similarities
from gensim.models import coherencemodel
import pyLDAvis.gensim




import logging
import warnings

from logging.handlers import RotatingFileHandler

logger_file_handler = RotatingFileHandler(u'working_data/logs/unsupervised_models.log')
logger_file_handler.setLevel(logging.INFO)

logging.captureWarnings(True)

logger = logging.getLogger(__name__)
warnings_logger = logging.getLogger("py.warnings")

logger.addHandler(logger_file_handler)
logger.setLevel(logging.DEBUG)
warnings_logger.addHandler(logger_file_handler)



def logmsg(msg):
    msg = str(msg)
    print(msg)
    logger.info(msg)
    return

def renamecols(df, dfname):
    dfcols = {}
    for column in df.columns:
        if column != 'unique_photo_id':
            dfcols[column] = dfname+'_'+column
        
    df = df.rename(columns=dfcols)
    return df
    


def cleanuniqueid(unique_photo_id):
    return unique_photo_id.split('.')[0].replace(' ','')

def dummify(col, confidence = 0):
    if col > confidence:
        return 1
    else:
        return 0

def renametags(tag, dfname):
    return   dfname + '_' + str(tag) 


def create_tags(data, likelihood, traintest, freqthres=None):
    
    
    tags = pd.DataFrame()
    if 'google_classifier' in data.columns:
        df = data[data.google_classifier == 'google_label_detection'][['unique_photo_id','google_label_description','google_label_score']]
        df = df.rename(columns={'google_label_description': 'label', 'google_label_score': 'likelihood'})
        df['label'] = df['label'].apply(renametags, args=('google',))
        
        if freqthres:
            tagfreq = pd.DataFrame(df['label'].value_counts()).reset_index().rename(columns={'label': 'count', 'index': 'label'})
            tagfreq['freq'] = tagfreq['count'] / len(df['unique_photo_id'].unique().tolist())
            df = df[df['label'].isin(tagfreq[tagfreq['freq'] >= freqthres]['label'].values.tolist())]
        
        tags = tags.append(df)

    if 'clarifai_label' in data.columns:
        df = data[data.classifier == 'clarifai'][['unique_photo_id','clarifai_label','clarifai_likelihood_value']]
        df = df.rename(columns={'clarifai_label': 'label', 'clarifai_likelihood_value': 'likelihood'})
        df['label'] = df['label'].apply(renametags, args=('clarifai',))
        
        if freqthres:
            tagfreq = pd.DataFrame(df['label'].value_counts()).reset_index().rename(columns={'label': 'count', 'index': 'label'})
            tagfreq['freq'] = tagfreq['count'] / len(df['unique_photo_id'].unique().tolist())
            df = df[df['label'].isin(tagfreq[tagfreq['freq'] >= freqthres]['label'].values.tolist())]
        
        tags = tags.append(df)
        
    if 'microsoft_category_label' in data.columns:
        df = data[data.classifier == 'microsoft_tags'][['unique_photo_id','microsoft_tags_name','microsoft_tags_score']]
        df = df.rename(columns={'microsoft_tags_name': 'label', 'microsoft_tags_score': 'likelihood'})
        df['label'] = df['label'].apply(renametags, args=('microsoft_tag',))
        
        if freqthres:
            tagfreq = pd.DataFrame(df['label'].value_counts()).reset_index().rename(columns={'label': 'count', 'index': 'label'})
            tagfreq['freq'] = tagfreq['count'] / len(df['unique_photo_id'].unique().tolist())
            df = df[df['label'].isin(tagfreq[tagfreq['freq'] >= freqthres]['label'].values.tolist())]
        
        tags = tags.append(df)

        df = data[data.classifier == 'microsoft_category'][['unique_photo_id','microsoft_category_label','microsoft_category_score']]
        df = df.rename(columns={'microsoft_category_label': 'label', 'microsoft_category_score': 'likelihood'})
        df['label'] = df['label'].apply(renametags, args=('microsoft_cat',))

        
        if freqthres:
            tagfreq = pd.DataFrame(df['label'].value_counts()).reset_index().rename(columns={'label': 'count', 'index': 'label'})
            tagfreq['freq'] = tagfreq['count'] / len(df['unique_photo_id'].unique().tolist())
            df = df[df['label'].isin(tagfreq[tagfreq['freq'] >= freqthres]['label'].values.tolist())]
        
        tags = tags.append(df)
    
    
    results = pd.DataFrame()
    for photo in traintest['unique_photo_id'].unique().tolist():
        tmp = tags[(tags['unique_photo_id'] == photo) & (tags['likelihood'] >= likelihood)]
        image_labels = []
        for label in tmp['label'].values.tolist():
            image_labels.append(label)
        
        res = {}
        res['unique_photo_id'] = photo
        res['image_labels'] = image_labels
        
        results = results.append(pd.DataFrame([res,]))
        
        
        
    return results


def create_features_unsupervised(likelihoods=[0.9,], freqthresholds=[0,]):
    logmsg('Starting with unsupervised machine learning')
    logmsg('Creating features dataset')

    traintest = pd.read_pickle('working_data/manualcoding/manualcoding_traintest.pkl')
    logmsg('loaded manual coding of subsample. N = ' + str(len(traintest)))

    traintest_unique_ids =  traintest['unique_photo_id'].unique().tolist()
    logmsg('total of unique IDs in subsample. N = ' + str(len(traintest_unique_ids)))



    # Loading machine coding
    google = pd.read_pickle('working_data/machinecoding/google_parsed.pkl')
    clarifai = pd.read_pickle('working_data/machinecoding/clarifai_parsed.pkl')
    microsoft = pd.read_pickle('working_data/machinecoding/microsoft_parsed.pkl')

    google['unique_photo_id'] = google['unique_photo_id'].apply(cleanuniqueid)
    clarifai['unique_photo_id'] = clarifai['unique_photo_id'].apply(cleanuniqueid)
    microsoft['unique_photo_id'] = microsoft['unique_photo_id'].apply(cleanuniqueid)


    # Ensuring alignment between manual and machine coding

    clarifai = clarifai[clarifai['unique_photo_id'].isin(traintest_unique_ids)]
    google = google[google['unique_photo_id'].isin(traintest_unique_ids)]
    microsoft = microsoft[microsoft['unique_photo_id'].isin(traintest_unique_ids)]

    google = renamecols(google, 'google')


    logmsg('Clarifai, Google and Microsoft datasets now aligned with train/test dataset per unique id')

    classifiers = {'google': google, 'clarifai': clarifai, 'microsoft': microsoft, 
               'allclassifiers': google.append(clarifai).append(microsoft)}


    datasets = {}
    counter = 0
    i = 0
    logmsg('creating datasets for each of the classifier, likelihoods and frequency thresholds. total combinations = ' + str(len(classifiers.items())*len(likelihoods)*len(freqthresholds)))
    for classifier, df in classifiers.items():
        for likelihood in likelihoods:
            for freqthres in freqthresholds:
                logmsg(str(counter+1)+' - ' + classifier+'_l'+str(likelihood)+'_f'+str(freqthres))
                datasets[classifier+'_l'+str(likelihood)+'_f'+str(freqthres)] = create_tags(df, likelihood, traintest, freqthres)
                counter += 1
                i+= 1

                if i == 10:
                    pickle.dump(datasets, open('working_data/unsupervised/datasets_topics.pickle','wb'))
                    i = 0

    pickle.dump(datasets, open('working_data/unsupervised/datasets_topics.pickle','wb'))

    return


def configure_lda(datasetname, num_topics, alpha, passes=50):
    p = {}
    p['lda_params']      = {'num_topics': num_topics, 'passes': passes, 'alpha': alpha}
    p['corpus_filename'] = 'working_data/unsupervised/topics/corpus_'+datasetname+'.mm'
    p['dict_filename']   = 'working_data/unsupervised/topics/dic_'+datasetname+'.dict'
    p['lda_filename']    = 'working_data/unsupervised/topics/model_'+datasetname+'_'+str(num_topics)+ '_' + str(alpha)+'.lda'
    
    return p

def create_model_visualize(datasetname, num_topics, alpha, data, results):

    
    try:

        # 1. Creating dictionary and corpus
        dic_labels = corpora.Dictionary(data.image_labels.values.tolist())
        dic_labels.compactify()
        dic_labels.save('working_data/unsupervised/topics/dic_'+ datasetname+'.dict')
        corpus_labels = [dic_labels.doc2bow(doc) for doc in data.image_labels.values.tolist()]
        corpora.MmCorpus.serialize('working_data/unsupervised/topics/corpus_'+ datasetname+'.mm', corpus_labels)    

        # 2. Creating topic model
        config = configure_lda(datasetname, num_topics, alpha)
        corpus = corpora.MmCorpus(config['corpus_filename'])
        dictionary = corpora.Dictionary.load(config['dict_filename'])
        lda = models.LdaModel(corpus, id2word=dictionary,
                            num_topics=config['lda_params']['num_topics'],
                            passes=config['lda_params']['passes'],
                            alpha = config['lda_params']['alpha'])

        logmsg(lda.print_topics())

        lda.save(config['lda_filename'])

        # 3. Creating and saving LDA visualizations
        corpus = corpora.MmCorpus('working_data/unsupervised/topics/corpus_'+datasetname+'.mm')
        dictionary = corpora.Dictionary.load('working_data/unsupervised/topics/dic_'+datasetname+'.dict')
        lda = models.LdaModel.load('working_data/unsupervised/topics/model_'+datasetname+'_'+str(num_topics)+ '_' + str(alpha)+'.lda')
        topic_data =  pyLDAvis.gensim.prepare(lda, corpus, dictionary)

        # pyLDAvis.enable_notebook()

        # pyLDAvis.gensim.prepare(lda, corpus, dictionary)

        topic_vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
        pyLDAvis.save_html(topic_vis, open('working_data/unsupervised/topics/' + datasetname+'_' + str(num_topics) + '_' + str(alpha) + '.html', 'w'))

        # 4. Storing topic metrics

        cm = coherencemodel.CoherenceModel(model=lda, texts=data.image_labels.values.tolist(), coherence='c_v')
        overallcoherence = cm.get_coherence()

        coherences = [item for item in cm.get_coherence_per_topic()]
        topics = {}
        for item in lda.print_topics(num_topics=num_topics):
            topics[item[0]] = item[1]



        counter = 0
        while counter < num_topics:
            res = {}
            res['dataset'] = datasetname
            res['num_topics'] = num_topics
            res['alpha'] = alpha
            res['overall_coherence'] = overallcoherence
            # Error handling
            try:
                res['topic'] = counter
                res['topic_words'] = topics[counter]
            except:
                res['topic_error'] = counter
            res['topic_coherence'] = coherences[counter]
            results = results.append(pd.DataFrame([res,]))
            counter += 1
        logmsg('\n\n\n')
    except Exception as e:
        res = {}
        res['general_error'] = str(e)
        res['dataset'] = datasetname
        res['num_topics'] = num_topics
        res['alpha'] = alpha
        results = results.append(pd.DataFrame([res,]))
    
    results.to_csv('working_data/unsupervised/topics/_results.csv', index=False)
    results.to_pickle('working_data/unsupervised/topics/_results.pkl')
    return results


def create_unsupervised_LDA_models(num_topics_list=[20,], alphas=[0.05,] ):
    try: 
        results = pd.read_pickle('working_data/unsupervised/topics/_results.pkl')
        completed = []
        for datasetname, alpha, num_topics in results[['dataset', 'alpha', 'num_topics']].values.tolist():
            completed.append(datasetname+'_'+str(alpha)+'_'+str(num_topics))
    except:
        results = pd.DataFrame()
        completed = []


    datasets = pickle.load(open('working_data/unsupervised/datasets_topics.pickle','rb'))
    logmsg('starting with LDA topic model creation')
    combinations = len(datasets.items()) * len(num_topics_list) * len(alphas)

    logmsg('creating datasets for each of the classifier, likelihoods, frequency thresholds, number of topics and alphas')
    logmsg('total combinations = ' + str(combinations))

    if len(results) > 0:
        counter = len(results[['alpha', 'dataset', 'num_topics']].drop_duplicates())
    else:
        counter = 0
    for datasetname, data in datasets.items():
        for num_topics in num_topics_list:
            for alpha in alphas:
                if datasetname+'_'+str(alpha)+'_'+str(num_topics) in completed:
                    pass
                else:
                    try:
                        logmsg(str(counter+1) + ' - ' + datasetname + ' num_topics: ' + str(num_topics) + ' alpha:  ' + str(alpha))
                        results = create_model_visualize(datasetname, num_topics, alpha, data, results)
                    except Exception as e:
                        res = {}
                        res['general_error'] = str(e)
                        res['dataset'] = datasetname
                        res['num_topics'] = num_topics
                        res['alpha'] = alpha
                        results = results.append(pd.DataFrame([res,]))
                        results.to_csv('working_data/unsupervisedtopics/_results.csv', index=False)
                        results.to_pickle('working_data/unsupervised/topics/_results.pkl')
                    counter += 1
    return


def check_equality(row, topicnames):
    # Setting cases in which there is no difference across topics to 99 - non-existing topic
    
    values = []
    for topic in topicnames:
        values.append(row[topic])
        
    if len(set(values)) == 1:
        row['no_predictions'] = True
        row['maxtopic'] = 99
    else:
        row['no_predictions'] = False
    return row

def execute_pipeline_unsupervised(features, variable, pipeline, tuned_parameters, scoring, train_set, test_set, classifier_name):
    logmsg('executing gridsearch')
    # GridSearch and best model selection

    clf = GridSearchCV(pipeline, tuned_parameters, scoring= scoring)

    clf.fit(train_set[features], train_set[variable])
    best_clf = clf.best_estimator_


    logmsg('best model found, saving predictions and model')
    # Saving the model
    joblib.dump(best_clf, 'working_data/unsupervised/models/'+classifier_name+'.joblibpickle') 
    # Making predictions
    test_set[variable+'_predicted_'+classifier_name] = best_clf.predict(test_set[features])
    test_set.to_pickle('working_data/predictions/'+classifier_name+'.pkl')

    return

def process_dataset(datasetname, dataset, variables, num_topics, alpha, traintest, overall_coherence, unsupervised_models_config):
    # Loading corpus, dictionary and LDA model associated with dataset
    corpus = corpora.MmCorpus('working_data/unsupervised/topics/corpus_'+datasetname+'.mm')
    dictionary = corpora.Dictionary.load('working_data/unsupervised/topics/dic_'+datasetname+'.dict')
    lda = models.LdaModel.load('working_data/unsupervised/topics/model_'+datasetname+'_'+str(num_topics)+'_' + str(alpha) + '.lda')
    
    
    
    # Creating and storing predictions
    topics_full = lda.inference(corpus)
    dataset = dataset.reset_index()
    dataset = dataset.reset_index()
    
    topics_full = pd.DataFrame(topics_full[0])
    topics_full = topics_full.reset_index()
    topics_full = topics_full.rename(columns={'index': 'level_0'})
    topicnames = [item for item in topics_full.columns if type(item) == int]
    dataset = dataset.merge(topics_full)
    
    # Identifying topic for the document based on highest loading
    max_topic = dataset[topicnames].idxmax(axis=1)
    max_topic = pd.DataFrame(max_topic).reset_index().rename(columns={'index': 'level_0', 0: 'maxtopic'})
    dataset = dataset.merge(max_topic)
    
    dataset = dataset.apply(check_equality, axis=1, args=(topicnames,))
    
    ## Dropping missing values
    dataset = dataset.dropna(subset=['maxtopic'])
    
    # Adding manual coding targets
    dataset = traintest.merge(dataset)

    # Performing NB
    train_set = dataset[dataset['traintest']=='train']
    test_set = dataset[dataset['traintest']=='test']

    for model in unsupervised_models_config['models']:
        for scoring in unsupervised_models_config['scoring_parameters']:
            for variable in variables:
                features = ['maxtopic']
                pipeline = model['pipeline']
                tuned_parameters = model['tuned_parameters']
                classifier_name = 'unsupervised_' + datasetname+'_'+str(num_topics)+'_' + str(alpha) + '_' + model['model_name'] + '_' + variable + '_' + scoring
                logmsg('Starting ' + classifier_name)
                try:
                    execute_pipeline_unsupervised(features, variable, pipeline, tuned_parameters, scoring, train_set, test_set, classifier_name)
                    calculate_precision_recall(classifier_name+'.pkl', [variable,], classifier_name)
                except Exception as e:
                    logmsg('error: ' + classifier_name)
                    logmsg(str(e))
    return



def select_unsupervised_models(variables, unsupervised_models_config, multiclass=False):
    logmsg('Using results of unsupervised LDA models to predict categories')

    traintest = pd.read_pickle('working_data/manualcoding/manualcoding_traintest.pkl')
    logmsg('loaded manual coding of subsample. N = ' + str(len(traintest)))

    
    unsupervised_results = pd.read_pickle('working_data/unsupervised/topics/_results.pkl')
    logmsg('loading results of LDA topic models. Total models = ' + str(len(unsupervised_results[['alpha', 'dataset', 'num_topics']].drop_duplicates())))
    logmsg('removing models that had an error message. N = ' + str(len(unsupervised_results[unsupervised_results['general_error'].isnull()==False][['alpha', 'dataset', 'num_topics']].drop_duplicates())))
    unsupervised_results = unsupervised_results[unsupervised_results['general_error'].isnull()]
    logmsg('removing models with overal coherence = 1. N = ' + str(len(unsupervised_results[unsupervised_results['overall_coherence']==1][['alpha', 'dataset', 'num_topics']].drop_duplicates())))
    unsupervised_results = unsupervised_results[unsupervised_results['overall_coherence']!= 1]
    logmsg('Total models = ' + str(len(unsupervised_results[['alpha', 'dataset', 'num_topics']].drop_duplicates())))    

    unsupervised_unique = unsupervised_results[['dataset', 'alpha', 'num_topics', 'overall_coherence']].drop_duplicates()
    unsupervised_unique = unsupervised_unique.sort_values(by='overall_coherence', ascending=False)
    logmsg('Top 10 models by overall coherence')
    logmsg(unsupervised_unique.head(10))


    datasets = pickle.load(open('working_data/unsupervised/datasets_topics.pickle','rb'))
    logmsg('loading datasets used for LDA topic models. N = ' + str(len(datasets.items())))


    for datasetname, alpha, num_topics, overall_coherence in unsupervised_unique[['dataset', 'alpha', 'num_topics', 'overall_coherence']].values.tolist():
        dataset = datasets[datasetname]
        logmsg(datasetname + ' - alpha: ' + str(alpha) +  ' - num_topics: ' + str(num_topics) + ' - overall coherence: ' + str(overall_coherence))
        process_dataset(datasetname, dataset, variables, num_topics, alpha, traintest, overall_coherence, unsupervised_models_config)


    return






