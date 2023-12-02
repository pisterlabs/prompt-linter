from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn import metrics, datasets
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn import svm
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import FeatureUnion
import logging as logger
import pandas as pd
from util_tools import globals
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from time import time
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import gensim
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
#import keras
import pickle

logger.basicConfig(level=logger.INFO, filename="/Users/emrecalisir/git/brexit/CSSforPolitics/logs/predictor.log", format="%(asctime)s %(message)s")



###########################################################
#######In this file, there are only the ml methods ########
###########that may be used only once or more##############
###########################################################


def predict_with_multiple_classifiers(classifier, df_unlabeled, model_remain, model_leave):
    try:
        logger.info("started to prediction")
        records = df_unlabeled[globals.PROCESSED_TEXT_COLUMN].tolist()
        y_preds_remain= model_remain.predict_proba(records)[:, 1]
        y_preds_leave = model_leave.predict_proba(records)[:, 1]
        df_unlabeled["y_preds_remain"] = pd.Series(y_preds_remain)
        df_unlabeled["y_preds_leave"] = pd.Series(y_preds_leave)
        print("ok")
    except Exception as ex:
        logger.error(str(ex))

    return df_unlabeled


def predict_with_remain_classifier(processed_text, model_remain):
    remain_confidence= model_remain.predict_proba(processed_text)[:, 1]
    return remain_confidence


def predict_with_leave_classifier(processed_text, model_leave):
    leave_confidence= model_leave.predict_proba(processed_text)[:, 1]
    return leave_confidence


def predict_unlabeled_data(is_binary_classification, classifier, df_train, df_unlabeled, remove_low_pred=False, prob_enabled=False):
    try:
        logger.info("started to prediction")

        if not is_binary_classification:
            classifier.fit(df_train[globals.PROCESSED_TEXT_COLUMN].tolist(), df_train[globals.TARGET_COLUMN].tolist())
            if prob_enabled:
                y_unlabeled_pair = classifier.predict_proba(df_unlabeled[globals.PROCESSED_TEXT_COLUMN].tolist())
                y_unlabeled = y_unlabeled_pair[:, 1]
            else:
                y_unlabeled = classifier.predict(df_unlabeled[globals.PROCESSED_TEXT_COLUMN].tolist())

            df_unlabeled["pred"]=pd.Series(y_unlabeled)
            logger.info("completed prediction. ")
            logger.info(df_unlabeled["pred"].value_counts())
            return df_unlabeled
            #if remove_low_pred:
            #    y_test, y_pred = discard_low_pred_prob_prediction_couple(x_unlabeled, y_unlabeled)
    except Exception as ex:
        logger.error(str(ex))


def build_lda_model(corpus, id2word, topic_cnt):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=topic_cnt,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    return lda_model


def evaluate_lda_results(corpus, id2word, texts, lda_model, topic_cnt, filename_read, visual_enabled = True):
    top_topics = lda_model.top_topics(corpus=corpus)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / topic_cnt
    logger.info("Average topic coherence: " + str(avg_topic_coherence))

    logger.info("top topics ordered by coherence score")
    logger.info(str(top_topics))

    # Print the Keyword in the 10 topics
    logger.info("topics: " + str(lda_model.print_topics()))

    logger.info("completed operations")

    # Print the Keyword in the 10 topics
    logger.info(lda_model.print_topics())
    logger.info("topics: : " + str(lda_model.print_topics()))

    # mallet operations... comment line because takes error in ubuntu
    # logger.info("ldamallet topics: " + ldamallet.show_topics(formatted=False))
    # Compute Coherence Score
    # coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word,coherence='c_v')
    # coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    # logger.info('\nldamallet Coherence Score: ', coherence_ldamallet)

    # Compute Perplexity
    logger.info("Perplexity: %s", lda_model.log_perplexity(corpus))

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    logger.info("Coherence Score: " + str(coherence_lda))

    # Visualize the topics
    # pyLDAvis.enable_notebook()
    if visual_enabled:
        vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        lda_file = filename_read + "_LDA_Visualization.html"
        pyLDAvis.save_html(vis, lda_file)


def find_best_parameters(parameters, pipeline, X, y):
    #parameter_searcher = GridSearchCV(pipeline, parameters, cv=5, n_jobs=2, verbose=1)
    n_iter_search = 20
    parameter_searcher = RandomizedSearchCV(pipeline, param_distributions=parameters,
                                       n_iter=n_iter_search)
    logger.info("Performing grid search...")
    logger.info("pipeline:", [name for name, _ in pipeline.steps])
    logger.info("parameters:")
    logger.info(parameters)
    t0 = time()
    parameter_searcher.fit(X, y)
    logger.info("done in %0.3fs" % (time() - t0))

    logger.info("Best score: %0.3f" % parameter_searcher.best_score_)
    logger.info("Best parameters set:")
    best_parameters = parameter_searcher.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        logger.info("\t%s: %r" % (param_name, best_parameters[param_name]))
    logger.info("Completed grid search")


def get_model(model_name):
    if model_name == "svm-linear":
        return svm.SVC(kernel="linear", C=10)
    elif model_name == "svm-linear-prob":
        return svm.SVC(kernel="linear", C=10, probability=True)
    elif model_name == "svm-rbf":
        return svm.SVC(kernel='rbf', gamma=0.7, C=1)
    elif model_name == "xgboost":
        return XGBClassifier(seed=42)
    elif model_name == "rf":
        return RandomForestClassifier(n_estimators=20)
    elif model_name == "log":
        return LogisticRegression()
    elif model_name == "sgd":
        return SGDClassifier(**globals.SGD_BEST_PARAMS)


def get_pipeline(feature_type, vect, tfidf, clf):
    pipeline = None
    if feature_type == "single":
        pipeline = Pipeline([('vect', vect),
                             ('tfidf', tfidf),
                             ('clf', clf),
                             ])
    elif feature_type == "feature_union":
        pipeline = Pipeline([

            ('union', FeatureUnion(
                transformer_list=[
                    ('text_stats_pipe', Pipeline([
                        ('selector', ItemSelector(key=globals.PROCESSED_TEXT_COLUMN)),
                        ('text_stats', TextStatsTransformer()),
                        ('special_keywords', VipTransformer()),
                    ])),
                    ('ngram_tf_idf', Pipeline([
                        ('selector', ItemSelector(key=globals.PROCESSED_TEXT_COLUMN)),
                        ('vect', vect),
                        ('tf_idf', tfidf)
                    ]))
                ],

                # weight components in FeatureUnion
                transformer_weights={
                    'text_stats_pipe': 1,
                    'ngram_tf_idf': 0.3
                }
            )),
            ('clf', clf)
        ]
        )
    return pipeline


def draw_confusion_matrix(y_test, y_pred):
    # Model Evaluation here
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 16}
    plt.rc('font', **font)
    plt.interactive(False)
    conf_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=[1, 2],
                yticklabels=[1, 2])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=True)


def print_confusion_matrix(is_binary_classification, y_test, y_pred):
    if is_binary_classification:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        logger.info("tn:" + str(tn) + " fn:" + str(fn) + " tp:" + str(tp) + " fp:" + str(fp))
    else:
        logger.info(confusion_matrix(y_test, y_pred))


def run_and_evaluate_cross_validation(is_binary_classification, is_scaling_enabled, classifier, X, y, is_plot_enabled=False):
    if is_scaling_enabled:
        X = scale_X(X)
    X, y = shuffle(X, y)
    cross_val_scores = cross_val_score(classifier, X, y, cv=10)
    logger.info(str(cross_val_scores))
    logger.info("cross val score: " + str(cross_val_scores.mean()))
    y_pred = cross_val_predict(classifier, X, y, cv=10)
    #print_false_predicted_entries(X, y_pred, y)
    logger.info(metrics.classification_report(y, y_pred, target_names=None))
    print_confusion_matrix(is_binary_classification, y, y_pred)
    print_evaluation_stats(y, y_pred,False)

    if is_plot_enabled:
        draw_confusion_matrix(y, y_pred)

def cross_validate(model, x, y, folds=10, repeats=5):
    '''
    Function to do the cross validation - using stacked Out of Bag method instead of averaging across folds.
    model = algorithm to validate. Must be scikit learn or scikit-learn like API (Example xgboost XGBRegressor)
    x = training data, numpy array
    y = training labels, numpy array
    folds = K, the number of folds to divide the data into
    repeats = Number of times to repeat validation process for more confidence
    '''
    try:
        ypred = np.zeros((len(y),repeats))
        score = np.zeros(repeats)
        x = np.array(x)
        for r in range(repeats):
            i=0
            print('Cross Validating - Run', str(r + 1), 'out of', str(repeats))
            x,y = shuffle(x,y,random_state=r) #shuffle data before each repeat
            kf = KFold(n_splits=folds,random_state=i+1000) #random split, different each time
            for train_ind,test_ind in kf.split(x):
                print('Fold', i+1, 'out of',folds)
                xtrain,ytrain = x[train_ind,:],y[train_ind]
                xtest,ytest = x[test_ind,:],y[test_ind]
                model.fit(xtrain, ytrain)
                ypred[test_ind,r]=model.predict(xtest)
                i+=1
            score[r] = R2(ypred[:,r],y)
        print('\nOverall R2:',str(score))
        print('Mean:',str(np.mean(score)))
        print('Deviation:',str(np.std(score)))

    except Exception as ex:
        logger.error(str(ex))

def R2(ypred, ytrue):
    y_avg = np.mean(ytrue)
    SS_tot = np.sum((ytrue - y_avg)**2)
    SS_res = np.sum((ytrue - ypred)**2)
    r2 = 1 - (SS_res/SS_tot)
    return r2


def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def restore_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def run_prob_based_train_test_kfold_roc_curve_plot(classifier, tweet_ids, x, y, is_plot_enabled=True, discard_low_pred=False):
    min_discard_prob=0.3
    max_discard_prob=0.7
    n_splits = 10
    #y = keras.utils.to_categorical(y, 3)
    y = label_binarize(y, classes=[0, 1])
    x, y = shuffle(x, y)
    cv = StratifiedKFold(n_splits=n_splits)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    #y = label_binarize(y, classes=[0, 1])
    i = 0
    logger.info("###" + str(n_splits) + "-fold started ###")
    cum_f1_score = 0
    try:
        cnt = 0
        counter_discarded_sum = 0
        counter_sum = 0
        logger.info("size of x: " + str(len(x)))
        list_tweet_ids = []
        for train, test in cv.split(x, y):

            logger.info("## fold: " + str(i+1) + "started")

            x = np.array(x)
            y = np.array(y)
            tweet_ids_np = np.array(tweet_ids)
            X_train = x[train]
            y_train = y[train]
            X_test = x[test]
            y_test = y[test]

            tweet_ids_np_train = tweet_ids_np[train]
            tweet_ids_np_test = tweet_ids_np[test]


            classifier.fit(X_train, y_train)
            probas_ = classifier.predict_proba(X_test)

            # probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area the curve
            y_pred = probas_[:, 1]
            if discard_low_pred:
                counter_sum += len(y_test)
                y_test, y_pred, counter_discarded = discard_low_pred_prob_prediction_couple(y_test, y_pred, min_discard_prob, max_discard_prob)
                counter_discarded_sum += counter_discarded
            #print_false_predicted_entries(X_test, y_pred, y_test, True)
            cum_f1_score += print_evaluation_stats(y_test, y_pred, True)
            if discard_low_pred:
                y_pred_nd = np.asarray(y_pred)
                indices_of_pred, = np.where(y_pred_nd > 0.5)
            else:
                indices_of_pred, = np.where(y_pred > 0.5)
            print(type(indices_of_pred))
            indices_of_pred_list = indices_of_pred.tolist()
            tweet_ids_list = tweet_ids.tolist()
            tw_ids_stance = np.take(tweet_ids_list, indices_of_pred_list)
            tw_ids_stance_list = tw_ids_stance.tolist()
            for tw_id in tw_ids_stance_list:
                list_tweet_ids.append(tw_id)
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            if is_plot_enabled:
                plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            i += 1
            logger.info("## fold: " + str(i+1) + "completed")

        with open('/Users/emrecalisir/git/brexit/CSSforPolitics/tweets_stance.txt', 'w') as f:
            for item in list_tweet_ids:
                f.write("%s\n" % item)
        from datetime import date
        today = date.today()

        save_model(classifier, "/Users/emrecalisir/git/brexit/CSSforPolitics/models/LeaveClassifier_"+str(today))

        logger.info("Discarded: " + str(counter_discarded_sum) + " out of " + str(counter_sum) + " records")
        logger.info("Average weighted F1-score: " + str(cum_f1_score/n_splits))
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        logger.info("Mean AUC: " + str(mean_auc))
        if is_plot_enabled:
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Luck', alpha=.8)
            plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Stratified k-fold with k='+str(n_splits))
            plt.legend(loc="lower right")
            plt.show()
    except Exception as e:
        logger.error(e)


def run_and_evaluate_train_test(is_binary_classification, is_scaling_enabled, classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    classifier.fit(X_train, y_train)
    if is_binary_classification:
        logger.info("Natural TP rate=" + str(sum(y_test) / len(y_test)))

    if is_scaling_enabled:
        X_train, X_test = scale_train_test(X_train, X_test)

    logger.info(pd.Series(y_test).value_counts())
    y_pred = classifier.predict(X_test)
    logger.info("expected test results  :" + str(y_test))
    logger.info("predicted test results: " + str(y_pred))
    logger.info("accuracy score:" + str(accuracy_score(y_test, y_pred)))
    print_confusion_matrix(is_binary_classification, y_test, y_pred)
    logger.info(metrics.classification_report(y_test, y_pred, target_names=None))
    print_evaluation_stats(y_test, y_pred, False)
    draw_confusion_matrix(y_test, y_pred)
    print("ok")


def print_evaluation_stats(y_test, y_pred, is_prob_pred):
    if is_prob_pred:
        y_pred = [0 if x < 0.5 else 1 for x in y_pred]

    logger.info("Accuracy Score:" + str(accuracy_score(y_test, y_pred)))
    logger.info("Precision Score:" + str(precision_score(y_test, y_pred, average='weighted')))
    logger.info("Recall Score:" + str(recall_score(y_test, y_pred, average='weighted')))
    logger.info("F1 Score:" + str(f1_score(y_test, y_pred, average='weighted')))
    return f1_score(y_test, y_pred, average='weighted')


def tryy():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                             random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


def multiclass_roc(X_train, X_test, y_train, y_test, n_classes):
    # tryy()
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                             random_state=False))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    draw_plt(0, fpr, tpr, roc_auc)
    draw_plt(1, fpr, tpr, roc_auc)
    draw_plt(2, fpr, tpr, roc_auc)


def draw_plt(lw, fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def print_false_predicted_entries(inputs, predictions_prob, labels, is_prob_pred = False):

    if is_prob_pred:
        predictions = [0 if x < 0.5 else 1 for x in predictions_prob]
    else:
        predictions = predictions_prob
    counter_false_prediction = 0
    for input, prediction, label in zip(inputs, predictions, labels):
        if prediction != label:
            counter_false_prediction += 1
            logger.info("### " + input + ' ### has been classified as ' + str(prediction) + ' and should be '+ str(label))
    logger.info("total final test size: " + str(len(predictions)))
    logger.info("false predicted records size: " + str(counter_false_prediction))


def discard_low_pred_prob_prediction_class_1(y_test, y_pred, max_discard_prob):
    y_test_new = []
    y_pred_new = []
    counter_discarded = 0
    logger.info("total size of original entries: " + str(len(y_pred)))
    for i in range(0, len(y_pred)):
        pred_confidence = y_pred[i]

        if pred_confidence < max_discard_prob:
            logger.debug(str(i) + "th record pred prob: " + str(y_pred[i]) + ". It'll be discarded from evaluation part")
            counter_discarded += 1
            continue;
        y_pred_new.append(y_pred[i])
        y_test_new.append(y_test[i])
    logger.info("total size of discarded entries having low probability predictions scores: " + str(counter_discarded))

    return y_test_new, y_pred_new


def discard_low_pred_prob_prediction_couple(y_test, y_pred, min_discard_prob, max_discard_prob):
    y_test_new = []
    y_pred_new = []
    counter_discarded = 0
    logger.info("total size of original entries: " + str(len(y_pred)))

    for i in range(0, len(y_pred)):
        if y_pred[i] > min_discard_prob and y_pred[i] < max_discard_prob:
            logger.debug(str(i) + "th record pred prob: " + str(y_pred[i]) + ". It'll be discarded from evaluation part")
            counter_discarded += 1
            continue;
        y_pred_new.append(y_pred[i])
        y_test_new.append(y_test[i])
    logger.info("total size of discarded entries having low probability predictions scores: " + str(counter_discarded))

    return y_test_new, y_pred_new, counter_discarded


def run_prob_based_train_test_roc_curve_plot(is_binary_classification, is_scaling_enabled, classifier, X, y, remove_low_pred=False):
    y = label_binarize(y, classes=[0, 1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    if is_scaling_enabled:
        X_train, X_test = scale_train_test(X_train, X_test)

    if not is_binary_classification:
        multiclass_roc(X_train, X_test, y_train, y_test, 3)
    else:
        classifier.fit(X_train, y_train)
        if is_binary_classification:
            logger.info("Natural TP rate=" + str(sum(y_test) / len(y_test)))
        y_pred_pair = classifier.predict_proba(X_test)
        y_pred = y_pred_pair[:, 1]

        # logger.info('Top 100 first' + str(len(y_test)) + ' records in test dataset) -> ' + str(
        #    ratio(y_test, y_pred_prob, 1)))

        logger.info("test ended")
        logger.info('ROC AUC:', roc_auc_score(y_test, y_pred))
        # for i in [x * 0.1 for x in range(1, 6)]:
        #    i = round(i, 1)
        #    logger.info('Top' + str(int(i * 100)) + 'percentile = (first ' + str(
        #        int(i * len(y_test))) + ' records in test dataset) -> ' + str(
        #        ratio(y_test, y_pred_prob, pct=i)))

        is_roc_plot_enabled = True
        if is_roc_plot_enabled:
            plot_roc(y_test, y_pred)

        if remove_low_pred:
            y_test, y_pred = discard_low_pred_prob_prediction_couple(y_test, y_pred)

        new_list = [0 if x<0.5 else 1 for x in y_pred]
        print_false_predicted_entries(X_test, y_pred, y_test, True)
        print_evaluation_stats(y_test, y_pred, True)


def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    logger.info(grid_search.best_params_)
    return grid_search.best_params_


def scale_X(X):
    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    return scaled_X


def scale_train_test(X_train, X_test):
    scaler = MinMaxScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    return scaled_X_train, scaled_X_test


def plot_roc(y_test, preds):
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


