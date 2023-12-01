import os
import sys
sys.path.append("../")
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.width', 1000)
from prepare_datasets import *
from sklearn.model_selection import train_test_split
from create_model import CreateModel, MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from user_preferences import userPreferences
from care.care import CARE
from evaluate_counterfactuals import evaluateCounterfactuals
from recover_originals import recoverOriginals
from care.coherency_obj import coherencyObj
from utils import ord2theta
from itertools import permutations

def main():
    # defining path of data sets and experiment results
    path = '../'
    dataset_path = path + 'datasets/'
    experiment_path = path + 'experiments/'

    # defining the list of data sets
    datsets_list = {
        'adult': ('adult.csv', PrepareAdult, 'classification'),
        # 'diabetes': ('diabetes-sklearn', PrepareDiabetes, 'regression')
    }

    # defining the list of black-boxes
    blackbox_list = {
        'nn-c': MLPClassifier,
        'gb-c': GradientBoostingClassifier,
        # 'nn-r': MLPRegressor,
        # 'gb-r': GradientBoostingRegressor
    }

    # defining number of samples N and number of counterfactuals generated for every instance n_cf
    experiment_size = {
        'adult': (500, 1),
        'diabetes': (80, 1)
    }

    for dataset_kw in datsets_list:
        print('dataset=', dataset_kw)
        print('\n')

        # reading a data set
        dataset_name, prepare_dataset_fn, task = datsets_list[dataset_kw]
        dataset = prepare_dataset_fn(dataset_path,dataset_name)

        # splitting the data set into train and test sets
        X, y = dataset['X_ord'], dataset['y']
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for blackbox_name, blackbox_constructor in blackbox_list.items():
            print('blackbox=', blackbox_name)

            # creating black-box model
            blackbox = CreateModel(dataset, X_train, X_test, Y_train, Y_test, task, blackbox_name, blackbox_constructor)
            if blackbox_name == 'nn-c':
                predict_fn = lambda x: blackbox.predict_classes(x).ravel()
                predict_proba_fn = lambda x: np.asarray([1-blackbox.predict(x).ravel(), blackbox.predict(x).ravel()]).transpose()
            else:
                predict_fn = lambda x: blackbox.predict(x).ravel()
                predict_proba_fn = lambda x: blackbox.predict_proba(x)

            # setting experiment size for the data set
            N, n_cf = experiment_size[dataset_kw]

            # creating/opening a csv file for storing results
            exists = os.path.isfile(
                experiment_path + 'care_temporal_action_sequence_%s_%s_cfs_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            if exists:
                os.remove(experiment_path + 'care_temporal_action_sequence_%s_%s_cfs_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf))
            cfs_results_csv = open(
                experiment_path + 'care_temporal_action_sequence_%s_%s_cfs_%s_%s.csv' % (dataset['name'], blackbox_name, N, n_cf), 'a')

            # creating an instance of CARE explainer
            explainer = CARE(dataset, task=task, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn,
                             SOUNDNESS=True, COHERENCY=True, ACTIONABILITY=True,
                             n_cf=n_cf, n_population=200, n_generation=20)

            # fitting the explainer on the training data
            explainer.fit(X_train, Y_train)

            # explaining test samples
            explained = 0
            for x_ord in X_test:

                # set user preferences
                user_preferences = userPreferences(dataset, x_ord)

                # generating counterfactuals
                explanations = explainer.explain(x_ord, user_preferences=user_preferences)

                # extracting results
                cfs_ord = explanations['cfs_ord']
                toolbox = explanations['toolbox']
                objective_names = explanations['objective_names']
                featureScaler = explanations['featureScaler']
                feature_names = dataset['feature_names']

                # evaluating counterfactuals
                cfs_ord, \
                cfs_eval, \
                x_cfs_ord, \
                x_cfs_eval = evaluateCounterfactuals(x_ord, cfs_ord, dataset, predict_fn, predict_proba_fn, task,
                                                     toolbox, objective_names, featureScaler, feature_names)

                # recovering counterfactuals in original format
                x_org, \
                cfs_org, \
                x_cfs_org, \
                x_cfs_highlight = recoverOriginals(x_ord, cfs_ord, dataset, feature_names)

                # temporal action sequence construction
                cfs_sequences = []
                for i in range(n_cf):
                    changed_features = np.where(cfs_ord.to_numpy()[i, :] != x_ord)[0]
                    orders = permutations(list(changed_features), len(changed_features))
                    order_cost = {}
                    for o in list(orders):
                        corr_cost = []
                        cf_ord = x_ord.copy()
                        for f in list(o):
                            cf_ord[f] = cfs_ord.iloc[i, f]
                            corr = coherencyObj(x_ord, cf_ord, dataset['feature_width'], dataset['continuous_indices'],
                                                dataset['discrete_indices'], explainer.correlationModel)
                            corr_cost.append(corr)
                        order_cost[o] = np.mean(corr_cost)
                    cfs_sequences.append(order_cost)

                best_steps = [None] * n_cf
                worst_steps  = [None] * n_cf
                for i, d in enumerate(cfs_sequences):
                    best_val = np.inf
                    worst_val = -np.inf
                    for key, val in d.items():
                        if val < best_val:
                            best_steps[i] = {key: val}
                            best_val = val
                        if val > worst_val:
                            worst_steps[i] = {key: val}
                            worst_val = val

                # print counterfactuals and their corresponding objective values
                print('\n')
                print(x_cfs_highlight)
                print('\n')
                print(x_cfs_eval)
                print('\n')

                # finding best and worst temporal action sequences for every counterfactual using the coherency model
                print('Best temporal action sequence:')
                best_sequence = []
                best_sequence.append([None,None])
                for i, bs in enumerate(best_steps):
                    for key, val in bs.items():
                        order = [dataset['feature_names'][f] for f in key]
                        order = ['%s' % (order[i]) for i in range(len(order))]
                        order = ' > '.join(order)
                        best_sequence.append([order,val])
                        s = 'cf_'+ str(i) + ' | order: ' + str(order) + ' | cost: ' + str(val)
                        print(s)

                print('\n')
                print('Worst temporal action sequence:')
                worst_sequence = []
                worst_sequence.append([None, None])
                for i, ws in enumerate(worst_steps):
                    for key, val in ws.items():
                        order = [dataset['feature_names'][f] for f in key]
                        order = ['%s' % (order[i]) for i in range(len(order))]
                        order = ' > '.join(order)
                        worst_sequence.append([order,val])
                        s = 'cf_'+ str(i) + ' | order: ' + str(order) + ' | cost: ' + str(val)
                        print(s)

                # storing the counterfactuals along with the best and worst temporal action sequences
                best_sequence_df = pd.DataFrame(data=best_sequence, columns=['Best Sequence', 'Best Cost'])
                best_sequence_df = best_sequence_df.set_index(x_cfs_ord.index)
                worst_sequence_df = pd.DataFrame(data=worst_sequence, columns=['Worst Sequence', 'Worst Cost'])
                worst_sequence_df = worst_sequence_df.set_index(x_cfs_ord.index)

                cfs_results = pd.concat([x_cfs_highlight, best_sequence_df, worst_sequence_df], axis=1)
                cfs_results.to_csv(cfs_results_csv)
                cfs_results_csv.write('\n')
                cfs_results_csv.flush()

                explained += 1

                print('\n')
                print('-----------------------------------------------------------------------')
                print("%s|%s: %d/%d explained" % (dataset['name'], blackbox_name, explained, N))
                print('-----------------------------------------------------------------------')

                if explained == N:
                    break

            cfs_results_csv.close()

if __name__ == '__main__':
    main()
