import array
import pandas as pd
from math import *
from deap import algorithms, base, creator, tools
from utils import *
from care.outcome_obj import outcomeObj
from care.distance_obj import distanceObj
from care.sparsity_obj import sparsityObj
from care.proximity_obj import proximityObj
from care.connectedness_obj import connectednessObj
from care.actionability_obj import actionabilityObj
from care.coherency_obj import coherencyObj
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import f1_score, r2_score
from dython import nominal
import hdbscan
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CARE():
    def __init__(self,
                 dataset,
                 task='classification',
                 predict_fn=None,
                 predict_proba_fn=None,
                 SOUNDNESS=False,
                 COHERENCY=False,
                 ACTIONABILITY=False,
                 n_cf=5,
                 response_quantile=4,
                 K_nbrs=500,
                 corr_thresh=None,
                 corr_model_train_percent=0.8,
                 corr_model_score_thresh=0.7,
                 n_population='adaptive',
                 n_generation=20,
                 hof_size=100,
                 x_init=0.3,
                 neighbor_init=0.6,
                 random_init=1.0,
                 crossover_perc=0.6,
                 mutation_perc=0.3,
                 division_factor=5,
                 ):

        self.dataset = dataset
        self.feature_names =  dataset['feature_names']
        self.n_features = len(dataset['feature_names'])
        self.feature_width = dataset['feature_width']
        self.continuous_indices = dataset['continuous_indices']
        self.discrete_indices = dataset['discrete_indices']
        self.task = task
        self.predict_fn = predict_fn
        self.predict_proba_fn = predict_proba_fn
        self.SOUNDNESS = SOUNDNESS
        self.COHERENCY = COHERENCY
        self.ACTIONABILITY = ACTIONABILITY
        self.n_cf = n_cf
        self.response_quantile = response_quantile
        self.K_nbrs = K_nbrs
        self.corr_thresh = corr_thresh
        self.corr_model_train_percent = corr_model_train_percent
        self.corr_model_score_thresh = corr_model_score_thresh
        self.n_population = n_population
        self.n_generation = n_generation
        self.hof_size = hof_size
        self.init_probability = [x_init, neighbor_init, random_init] / np.sum([x_init, neighbor_init, random_init])
        self.crossover_perc = crossover_perc
        self.mutation_perc = mutation_perc
        self.division_factor = division_factor
        self.objectiveFunction = self.constructObjectiveFunction()

    def constructObjectiveFunction(self):

        print('Constructing objective function according to SOUNDNESS, COHERENCY, and ACTIONABILITY hyper-parameters ...')

        if self.SOUNDNESS == False and self.COHERENCY == False and self.ACTIONABILITY == False:
            # objective names
            self.objective_names = ['outcome', 'distance', 'sparsity']
            # objective weights, -1.0: cost function, 1.0: fitness function
            self.objective_weights = (-1.0, -1.0, -1.0)
            # number of objectives
            self.n_objectives = 3

            # defining objective function
            def objectiveFunction(x_ord, x_org, task, cf_class, cf_range, probability_thresh, proximity_model,
                                  connectedness_model, user_preferences, dataset, predict_fn, predict_proba_fn,
                                  feature_width, continuous_indices, discrete_indices, featureScaler,
                                  correlationModel, cf_theta):

                # constructing counterfactual from the EA decision variables
                cf_theta = np.asarray(cf_theta)
                cf_org = theta2org(cf_theta, featureScaler, dataset)
                cf_ord = org2ord(cf_org, dataset)
                cf_ohe = ord2ohe(cf_ord, dataset)

                # objective 1: desired outcome
                outcome_cost = outcomeObj(cf_ohe, task, predict_fn, predict_proba_fn,
                                             probability_thresh, cf_class, cf_range)
                # objective 2: feature distance
                distance_cost = distanceObj(x_ord, cf_ord, feature_width, continuous_indices, discrete_indices)

                # objective 3: sparsity
                sparsity_cost = sparsityObj(x_org, cf_org)

                return outcome_cost, distance_cost, sparsity_cost

            return objectiveFunction

        elif self.SOUNDNESS == False and self.COHERENCY == False and self.ACTIONABILITY == True:
            # objective names
            self.objective_names = ['outcome', 'actionability', 'distance', 'sparsity']
            # objective weights, -1.0: cost function, 1.0: fitness function
            self.objective_weights = (-1.0, -1.0, -1.0, -1.0)
            # number of objectives
            self.n_objectives = 4

            # defining objective function
            def objectiveFunction(x_ord, x_org, task, cf_class, cf_range, probability_thresh, proximity_model,
                                  connectedness_model, user_preferences, dataset, predict_fn, predict_proba_fn,
                                  feature_width, continuous_indices, discrete_indices, featureScaler,
                                  correlationModel, cf_theta):

                # constructing counterfactual from the EA decision variables
                cf_theta = np.asarray(cf_theta)
                cf_org = theta2org(cf_theta, featureScaler, dataset)
                cf_ord = org2ord(cf_org, dataset)
                cf_ohe = ord2ohe(cf_ord, dataset)

                # objective 1: desired outcome
                outcome_cost = outcomeObj(cf_ohe, task, predict_fn, predict_proba_fn,
                                             probability_thresh, cf_class, cf_range)

                # objective 2: actionable recourse
                actionability_cost = actionabilityObj(x_org, cf_org, user_preferences)

                # objective 3: feature distance
                distance_cost = distanceObj(x_ord, cf_ord, feature_width, continuous_indices, discrete_indices)

                # objective 4: sparsity
                sparsity_cost = sparsityObj(x_org, cf_org)

                return outcome_cost, actionability_cost, distance_cost, sparsity_cost

            return objectiveFunction

        elif self.SOUNDNESS == False and self.COHERENCY == True and self.ACTIONABILITY == False:
            # objective names
            self.objective_names = ['outcome', 'coherency', 'distance', 'sparsity']
            # objective weights, -1.0: cost function, 1.0: fitness function
            self.objective_weights = (-1.0, -1.0, -1.0, -1.0)
            # number of objectives
            self.n_objectives = 4

            # defining objective function
            def objectiveFunction(x_ord, x_org, task, cf_class, cf_range, probability_thresh, proximity_model,
                                  connectedness_model, user_preferences, dataset, predict_fn, predict_proba_fn,
                                  feature_width, continuous_indices, discrete_indices, featureScaler,
                                  correlationModel, cf_theta):

                # constructing counterfactual from the EA decision variables
                cf_theta = np.asarray(cf_theta)
                cf_org = theta2org(cf_theta, featureScaler, dataset)
                cf_ord = org2ord(cf_org, dataset)
                cf_ohe = ord2ohe(cf_ord, dataset)

                # objective 1: desired outcome
                outcome_cost = outcomeObj(cf_ohe, task, predict_fn, predict_proba_fn,
                                             probability_thresh, cf_class, cf_range)

                # objective 2: coherency
                coherency_cost = coherencyObj(x_ord, cf_ord, feature_width, continuous_indices,
                                              discrete_indices, correlationModel)

                # objective 3: feature distance
                distance_cost = distanceObj(x_ord, cf_ord, feature_width, continuous_indices, discrete_indices)

                # objective 4: sparsity
                sparsity_cost = sparsityObj(x_org, cf_org)

                return outcome_cost, coherency_cost, distance_cost, sparsity_cost

            return objectiveFunction

        elif self.SOUNDNESS == False and self.COHERENCY == True and self.ACTIONABILITY == True:
            # objective names
            self.objective_names = ['outcome', 'actionability', 'coherency', 'distance', 'sparsity']
            # objective weights, -1.0: cost function, 1.0: fitness function
            self.objective_weights = (-1.0, -1.0, -1.0, -1.0, -1.0)
            # number of objectives
            self.n_objectives = 5

            # defining objective function
            def objectiveFunction(x_ord, x_org, task, cf_class, cf_range, probability_thresh, proximity_model,
                                  connectedness_model, user_preferences, dataset, predict_fn, predict_proba_fn,
                                  feature_width, continuous_indices, discrete_indices, featureScaler,
                                  correlationModel, cf_theta):

                # constructing counterfactual from the EA decision variables
                cf_theta = np.asarray(cf_theta)
                cf_org = theta2org(cf_theta, featureScaler, dataset)
                cf_ord = org2ord(cf_org, dataset)
                cf_ohe = ord2ohe(cf_ord, dataset)

                # objective 1: desired outcome
                outcome_cost = outcomeObj(cf_ohe, task, predict_fn, predict_proba_fn,
                                             probability_thresh, cf_class, cf_range)

                # objective 2: actionable recourse
                actionability_cost = actionabilityObj(x_org, cf_org, user_preferences)

                # objective 3: coherency
                coherency_cost = coherencyObj(x_ord, cf_ord, feature_width, continuous_indices,
                                              discrete_indices, correlationModel)

                # objective 4: feature distance
                distance_cost = distanceObj(x_ord, cf_ord, feature_width, continuous_indices, discrete_indices)

                # objective 5: sparsity
                sparsity_cost = sparsityObj(x_org, cf_org)

                return outcome_cost, actionability_cost, coherency_cost, distance_cost, sparsity_cost

            return objectiveFunction

        elif self.SOUNDNESS == True and self.COHERENCY == False and self.ACTIONABILITY == False:
            # objective names
            self.objective_names = ['outcome', 'proximity', 'connectedness', 'distance', 'sparsity']
            # objective weights, -1.0: cost function, 1.0: fitness function
            self.objective_weights = (-1.0, 1.0, 1.0, -1.0, -1.0)
            # number of objectives
            self.n_objectives = 5

            # defining objective function
            def objectiveFunction(x_ord, x_org, task, cf_class, cf_range, probability_thresh, proximity_model,
                                  connectedness_model, user_preferences, dataset, predict_fn, predict_proba_fn,
                                  feature_width, continuous_indices, discrete_indices, featureScaler,
                                  correlationModel, cf_theta):

                # constructing counterfactual from the EA decision variables
                cf_theta = np.asarray(cf_theta)
                cf_org = theta2org(cf_theta, featureScaler, dataset)
                cf_ord = org2ord(cf_org, dataset)
                cf_ohe = ord2ohe(cf_ord, dataset)

                # objective 1: desired outcome
                outcome_cost = outcomeObj(cf_ohe, task, predict_fn, predict_proba_fn,
                                             probability_thresh, cf_class, cf_range)
                # objective 2: proximity
                proximity_fitness = proximityObj(cf_ohe, proximity_model)

                # objective 3: connectedness
                connectedness_fitness = connectednessObj(cf_ohe, connectedness_model)

                # objective 4: feature distance
                distance_cost = distanceObj(x_ord, cf_ord, feature_width, continuous_indices, discrete_indices)

                # objective 5: sparsity
                sparsity_cost = sparsityObj(x_org, cf_org)

                return outcome_cost, proximity_fitness, connectedness_fitness, distance_cost, sparsity_cost

            return objectiveFunction

        elif self.SOUNDNESS == True and self.COHERENCY == False and self.ACTIONABILITY == True:
            # objective names
            self.objective_names = ['outcome', 'actionability', 'proximity', 'connectedness', 'distance', 'sparsity']
            # objective weights, -1.0: cost function, 1.0: fitness function
            self.objective_weights = (-1.0, -1.0, 1.0, 1.0, -1.0, -1.0)
            # number of objectives
            self.n_objectives = 6

            # defining objective function
            def objectiveFunction(x_ord, x_org, task, cf_class, cf_range, probability_thresh, proximity_model,
                                  connectedness_model, user_preferences, dataset, predict_fn, predict_proba_fn,
                                  feature_width, continuous_indices, discrete_indices, featureScaler,
                                  correlationModel, cf_theta):

                # constructing counterfactual from the EA decision variables
                cf_theta = np.asarray(cf_theta)
                cf_org = theta2org(cf_theta, featureScaler, dataset)
                cf_ord = org2ord(cf_org, dataset)
                cf_ohe = ord2ohe(cf_ord, dataset)

                # objective 1: desired outcome
                outcome_cost = outcomeObj(cf_ohe, task, predict_fn, predict_proba_fn,
                                             probability_thresh, cf_class, cf_range)

                # objective 2: actionable recourse
                actionability_cost = actionabilityObj(x_org, cf_org, user_preferences)

                # objective 3: proximity
                proximity_fitness = proximityObj(cf_ohe, proximity_model)

                # objective 4: connectedness
                connectedness_fitness = connectednessObj(cf_ohe, connectedness_model)

                # objective 5: feature distance
                distance_cost = distanceObj(x_ord, cf_ord, feature_width, continuous_indices, discrete_indices)

                # objective 6: sparsity
                sparsity_cost = sparsityObj(x_org, cf_org)

                return outcome_cost, actionability_cost, proximity_fitness, connectedness_fitness, \
                       distance_cost, sparsity_cost

            return objectiveFunction

        elif self.SOUNDNESS == True and self.COHERENCY == True and self.ACTIONABILITY == False:
            # objective names
            self.objective_names = ['outcome', 'coherency', 'proximity', 'connectedness', 'distance', 'sparsity']
            # objective weights, -1.0: cost function, 1.0: fitness function
            self.objective_weights = (-1.0, -1.0, 1.0, 1.0, -1.0, -1.0)
            # number of objectives
            self.n_objectives = 6

            # defining objective function
            def objectiveFunction(x_ord, x_org, task, cf_class, cf_range, probability_thresh, proximity_model,
                                  connectedness_model, user_preferences, dataset, predict_fn, predict_proba_fn,
                                  feature_width, continuous_indices, discrete_indices, featureScaler,
                                  correlationModel, cf_theta):

                # constructing counterfactual from the EA decision variables
                cf_theta = np.asarray(cf_theta)
                cf_org = theta2org(cf_theta, featureScaler, dataset)
                cf_ord = org2ord(cf_org, dataset)
                cf_ohe = ord2ohe(cf_ord, dataset)

                # objective 1: desired outcome
                outcome_cost = outcomeObj(cf_ohe, task, predict_fn, predict_proba_fn,
                                             probability_thresh, cf_class, cf_range)

                # objective 2: coherency
                coherency_cost = coherencyObj(x_ord, cf_ord, feature_width, continuous_indices,
                                              discrete_indices, correlationModel)

                # objective 3: proximity
                proximity_fitness = proximityObj(cf_ohe, proximity_model)

                # objective 4: connectedness
                connectedness_fitness = connectednessObj(cf_ohe, connectedness_model)

                # objective 5: feature distance
                distance_cost = distanceObj(x_ord, cf_ord, feature_width, continuous_indices, discrete_indices)

                # objective 6: sparsity
                sparsity_cost = sparsityObj(x_org, cf_org)

                return outcome_cost, coherency_cost, proximity_fitness, connectedness_fitness, \
                       distance_cost, sparsity_cost

            return objectiveFunction

        elif self.SOUNDNESS == True and self.COHERENCY == True and self.ACTIONABILITY == True:
            # objective names
            self.objective_names = ['outcome', 'actionability', 'coherency', 'proximity', 'connectedness',
                                    'distance', 'sparsity']
            # objective weights, -1.0: cost function, 1.0: fitness function
            self.objective_weights = (-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0)
            # number of objectives
            self.n_objectives = 7

            # defining objective function
            def objectiveFunction(x_ord, x_org, task, cf_class, cf_range, probability_thresh, proximity_model,
                                  connectedness_model, user_preferences, dataset, predict_fn, predict_proba_fn,
                                  feature_width, continuous_indices, discrete_indices, featureScaler,
                                  correlationModel, cf_theta):

                # constructing counterfactual from the EA decision variables
                cf_theta = np.asarray(cf_theta)
                cf_org = theta2org(cf_theta, featureScaler, dataset)
                cf_ord = org2ord(cf_org, dataset)
                cf_ohe = ord2ohe(cf_ord, dataset)

                # objective 1: desired outcome
                outcome_cost = outcomeObj(cf_ohe, task, predict_fn, predict_proba_fn,
                                             probability_thresh, cf_class, cf_range)

                # objective 2: actionable recourse
                actionability_cost = actionabilityObj(x_org, cf_org, user_preferences)

                # objective 3: coherency
                coherency_cost = coherencyObj(x_ord, cf_ord, feature_width, continuous_indices,
                                              discrete_indices, correlationModel)

                # objective 4: proximity
                proximity_fitness = proximityObj(cf_ohe, proximity_model)

                # objective 5: connectedness
                connectedness_fitness = connectednessObj(cf_ohe, connectedness_model)

                # objective 6: feature distance
                distance_cost = distanceObj(x_ord, cf_ord, feature_width, continuous_indices, discrete_indices)

                # objective 7: sparsity
                sparsity_cost = sparsityObj(x_org, cf_org)

                return outcome_cost, actionability_cost, coherency_cost, proximity_fitness, connectedness_fitness,\
                       distance_cost, sparsity_cost

            return objectiveFunction


    def groundtruthData(self):

        print('Identifying correctly predicted training data for each class/quantile ...')

        # dict to save ground-truth data
        groundtruth_data = {}
        # convert ordinal data to one-hot encoding
        X_train_ohe = ord2ohe(self.X_train, self.dataset)

        # classification task
        if self.task is 'classification':
            # find the number of classes in data
            self.n_classes = np.unique(self.Y_train)
            # predict the label of X_train
            pred_train = self.predict_fn(X_train_ohe)
            # find correctly predicted samples
            groundtruth_ind = np.where(pred_train == self.Y_train)
            # predict the label of correctly predicted samples
            pred_groundtruth = self.predict_fn(X_train_ohe[groundtruth_ind])
            # collect the ground-truth data of every class
            for c in self.n_classes:
                c_ind = np.where(pred_groundtruth == c)
                c_ind = groundtruth_ind[0][c_ind[0]]
                groundtruth_data[c] = self.X_train[c_ind].copy()
            return groundtruth_data

        # regression task
        elif self.task is 'regression':
            # divide the response values into quantiles
            q = np.quantile(self.Y_train, q=np.linspace(0, 1, self.response_quantile))
            self.response_ranges = [[q[i], q[i + 1]] for i in range(len(q) - 1)]
            # predict the response of X_train
            pred_train = self.predict_fn(X_train_ohe)
            # find correctly predicted samples
            abs_error = np.abs(self.Y_train - pred_train)
            mae = np.mean(abs_error)
            groundtruth_ind = np.where(abs_error <= mae)
            # predict the response of correctly predicted samples
            pred_groundtruth = self.predict_fn(X_train_ohe[groundtruth_ind])
            # collect the ground-truth data of every quantile
            for r in range(self.response_quantile - 1):
                r_ind = np.where(np.logical_and(pred_groundtruth >= self.response_ranges[r][0],
                                                pred_groundtruth <= self.response_ranges[r][1]))
                r_ind = groundtruth_ind[0][r_ind[0]]
                groundtruth_data[r] = self.X_train[r_ind].copy()
            return groundtruth_data

        else:
            raise TypeError("The task is not supported! CARE works on 'classification' and 'regression' tasks.")

    def featureScaler(self):
        # creating a scaler for mapping features in equal range
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        feature_scaler.fit(self.X_train)
        return feature_scaler

    def proximityModel(self):

        print('Creating Local Outlier Factor (LOF) models for measuring proximity ...')

        # creating Local Outlier Factor models for modeling proximity
        lof_models = {}
        for key, data in self.groundtruthData.items():
            data_ohe = ord2ohe(data, self.dataset)
            lof_model = LocalOutlierFactor(n_neighbors=1, novelty=True, algorithm='ball_tree', metric='minkowski', p=2)
            lof_model.fit(data_ohe)

            lof_models[key] = lof_model
        return lof_models

    def connectednessModel(self):

        print('Creating HDBSCAN clustering models for measuring connectedness ...')

        # creating HDBSCAN models for modeling connectedness
        hdbscan_models = {}
        for key, data in self.groundtruthData.items():
            data_ohe = ord2ohe(data, self.dataset)
            hdbscan_model = hdbscan.HDBSCAN(min_samples=2, metric='minkowski', p=2, prediction_data=True,
                                            approx_min_span_tree=False, gen_min_span_tree=True).fit(data_ohe)
            hdbscan_models[key] = hdbscan_model
        return hdbscan_models

    def correlationModel(self):

        print('Creating correlation models for coherency-preservation ...')

        ## Constructing Correlation Models

        # Calculate the correlation/strength-of-association of features in data-set
        # with both categorical and continuous features using:
            # Spearman's Rho for continuous-continuous pairs
            # Correlation Ratio for categorical-continuous pairs
            # Cramer's V for categorical-categorical pairs
        corr_con_con = pd.DataFrame(data=self.X_train).corr(method='spearman') # to consider non-linear correlations
        corr_rest = nominal.associations(self.X_train, nominal_columns=self.discrete_indices, plot=False)['corr']
        plt.close('all')
        corr = corr_rest.copy()
        for k in self.continuous_indices:
            for l in self.continuous_indices:
                corr.iloc[k,l] = corr_con_con.iloc[k,l]

        # only consider the features that have correlation above the correlation threshold / mean correlation
        corr = corr.to_numpy()
        corr = abs(corr)
        corr[np.diag_indices(corr.shape[0])] = 0.0

        if self.corr_thresh is None:
            self.corr_thresh = {'spearmans_rho': [],
                               'correlation_ratio':[],
                               'cramers_v':[]}
            for k in range(corr.shape[0]):
                for l in range(corr.shape[1]):
                    if k==l:
                        pass
                    else:
                        if k in self.continuous_indices and l in self.continuous_indices:
                            self.corr_thresh['spearmans_rho'].append(corr[k,l])
                        elif (k in self.continuous_indices and l in self.discrete_indices) or \
                                (k in self.discrete_indices and l in self.continuous_indices):
                            self.corr_thresh['correlation_ratio'].append(corr[k,l])
                        elif (k in self.discrete_indices and l in self.discrete_indices):
                            self.corr_thresh['cramers_v'].append(corr[k,l])

            for key,val in self.corr_thresh.items():
                self.corr_thresh[key] = np.mean(val)

        corr_ = np.zeros(corr.shape)
        for k in range(corr.shape[0]):
            for l in range(corr.shape[1]):
                if k == l:
                    pass
                else:
                    if k in self.continuous_indices and l in self.continuous_indices:
                        rho = self.corr_thresh['spearmans_rho']
                    elif (k in self.continuous_indices and l in self.discrete_indices) or \
                            (k in self.discrete_indices and l in self.continuous_indices):
                        rho = self.corr_thresh['correlation_ratio']
                    elif (k in self.discrete_indices and l in self.discrete_indices):
                        rho = self.corr_thresh['cramers_v']

                    corr_[k,l] = 1 if corr[k,l] >= rho else 0

        ## creating correlation models
        val_point = int(self.corr_model_train_percent * len(self.X_train))
        scores = []
        correlation_models = []
        for f in range(len(corr_)):
            inputs = np.where(corr_[f, :] == 1)[0]
            if len(inputs) > 0:
                if f in self.discrete_indices:
                    model = DecisionTreeClassifier()
                    model.fit(self.X_train[0:val_point, inputs], self.X_train[0:val_point, f])
                    score = f1_score(self.X_train[val_point:, f], model.predict(self.X_train[val_point:, inputs]),
                                     average='weighted')
                    scores.append(score)
                    correlation_models.append({'feature': f, 'inputs': inputs, 'model': model, 'score': score})
                elif f in self.continuous_indices:
                    model = Ridge()
                    model.fit(self.X_train[0:val_point, inputs], self.X_train[0:val_point, f])
                    score = r2_score(self.X_train[val_point:, f], model.predict(self.X_train[val_point:, inputs]))
                    scores.append(score)
                    correlation_models.append({'feature': f, 'inputs': inputs, 'model': model, 'score': score})

        # select models that have prediction score above a threshold value
        selected_models = np.where(scores >= np.float64(self.corr_model_score_thresh))[0]
        correlation_models = [correlation_models[m] for m in selected_models]

        return correlation_models

    def neighborhoodModel(self):

        print('Creating neighborhood models for every class/quantile of correctly predicted training data ...')

        neighborhood_models = {}
        for key, data in self.groundtruthData.items():
            data_ohe = ord2ohe(data, self.dataset)
            K_nbrs = min(self.K_nbrs, len(data_ohe))
            neighborhood_model = NearestNeighbors(n_neighbors=K_nbrs, algorithm='ball_tree', metric='minkowski', p=2)
            neighborhood_model.fit(data_ohe)
            neighborhood_models[key] = neighborhood_model
        return neighborhood_models

    def fit(self, X_train, Y_train):

        print('Fitting the framework on the training data ...')

        self.X_train = X_train
        self.Y_train = Y_train

        self.groundtruthData = self.groundtruthData()
        self.featureScaler = self.featureScaler()
        self.proximityModel = self.proximityModel()
        self.connectednessModel = self.connectednessModel()
        self.correlationModel = self.correlationModel()
        self.neighborhoodModel = self.neighborhoodModel()

    # creating toolbox for optimization algorithm
    def setupToolbox(self, x_ord, x_org, x_theta, cf_class, cf_range, probability_thresh, user_preferences,
                     neighbor_theta, proximity_model, connectedness_model):

        print('Creating toolbox for the optimization algorithm ...')

        # initialization function
        def initialization(x_theta, neighbor_theta, n_features, init_probability):
            method = np.random.choice(['x', 'neighbor', 'random'], size=1, p=init_probability)
            if method == 'x':
                return list(x_theta)
            elif method == 'neighbor':
                idx = np.random.choice(range(len(neighbor_theta)), size=1)
                return list(neighbor_theta[idx].ravel())
            elif method == 'random':
                return list(np.random.uniform(0, 1, n_features))

        # creating toolbox
        toolbox = base.Toolbox()
        creator.create("FitnessMulti", base.Fitness, weights=self.objective_weights)
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)
        toolbox.register("evaluate", self.objectiveFunction, x_ord, x_org, self.task, cf_class, cf_range,
                         probability_thresh, proximity_model, connectedness_model, user_preferences, self.dataset,
                         self.predict_fn, self.predict_proba_fn, self.feature_width, self.continuous_indices,
                         self.discrete_indices, self.featureScaler, self.correlationModel)
        toolbox.register("attr_float", initialization, x_theta, neighbor_theta, self.n_features, self.init_probability)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20.0, indpb=1.0 / self.n_features)
        ref_points = tools.uniform_reference_points(len(self.objective_weights), self.division_factor)
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

        return toolbox

    # executing the optimization algorithm
    def runEA(self):

        print('Running NSGA-III optimization algorithm ...')

        # Initialize statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        hof = tools.HallOfFame(self.hof_size)
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "min", "avg", "max"

        pop = self.toolbox.population(n=self.n_population)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Compile statistics about the population
        record = stats.compile(pop)
        logbook.record(pop=pop, gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)

        # Begin the generational process
        for gen in range(1, self.n_generation):
            offspring = algorithms.varAnd(pop, self.toolbox, self.crossover_perc, self.mutation_perc)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population from parents and offspring
            pop = self.toolbox.select(pop + offspring, self.n_population)

            # Compile statistics about the new population
            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(pop=pop, gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)

        fronts = tools.emo.sortLogNondominated(pop, self.n_population)

        return fronts, pop, hof, record, logbook

    # explain instance using multi-objective counterfactuals
    def explain(self,
                x,
                cf_class='neighbor',
                probability_thresh=0.5,
                cf_quantile='neighbor',
                user_preferences=None
                ):

        print('Generating counterfactual explanations ...')

        x_ord = x
        x_ohe = ord2ohe(x, self.dataset)
        x_org = ord2org(x, self.dataset)
        x_theta = ord2theta(x, self.featureScaler)

        if self.task is 'classification':
            cf_range = None
            # finding the label of counterfactual instance
            if cf_class is 'opposite':
                x_class = self.predict_fn(x_ohe.reshape(1,-1))
                cf_target = 1 - x_class[0]
                cf_class = cf_target
            elif cf_class is 'neighbor':
                x_proba = self.predict_proba_fn(x_ohe.reshape(1,-1))
                cf_target = np.argsort(x_proba)[0][-2]
                cf_class = cf_target
            elif cf_class is 'strange':
                x_proba = self.predict_proba_fn(x_ohe.reshape(1,-1))
                cf_target = np.argsort(x_proba)[0][0]
                cf_class = cf_target
            else:
                cf_target = cf_class

        elif self.task is 'regression':
            cf_class = None
            # finding the response range of counterfactual instance
            if cf_quantile is 'neighbor':
                x_response = self.predict_fn(x_ohe.reshape(1, -1))
                for i in range(len(self.response_ranges)):
                    if self.response_ranges[i][0] <= x_response <= self.response_ranges[i][1]:
                        x_quantile = i
                        break
                if x_quantile == 0:
                    cf_target = x_quantile + 1
                    cf_range = self.response_ranges[cf_target]
                elif x_quantile == self.response_quantile - 2:
                    cf_target = self.response_quantile - 3
                    cf_range = self.response_ranges[cf_target]
                else:
                    cf_target = x_quantile + 1
                    cf_range = self.response_ranges[cf_target]
            else:
                cf_target = cf_quantile
                cf_range = self.response_ranges[cf_target]

        # finding the neighborhood data of the counterfactual instance
        distances, indices = self.neighborhoodModel[cf_target].kneighbors(x_ohe.reshape(1, -1))
        neighbor_data = self.groundtruthData[cf_target][indices[0]].copy()
        neighbor_theta = self.featureScaler.transform(neighbor_data)

        # creating toolbox for the counterfactual instance
        proximity_model = self.proximityModel[cf_target]
        connectedness_model = self.connectednessModel[cf_target]
        self.toolbox = self.setupToolbox(x_ord, x_org, x_theta, cf_class, cf_range, probability_thresh,
                                         user_preferences, neighbor_theta, proximity_model, connectedness_model)

        # computing the number of population adaptively
        if self.n_population == 'adaptive':
            n_reference_points = factorial(self.n_objectives + self.division_factor - 1) / \
                                 (factorial(self.division_factor) * factorial(self.n_objectives - 1))
            self.n_population = int(n_reference_points + (4 - n_reference_points % 4))

        # running optimization algorithm for finding counterfactual instances
        fronts, pop, hof, record, logbook = self.runEA()

        ## constructing counterfactuals
        cfs_theta = np.asarray([i for i in hof.items])
        cfs_org = theta2org(cfs_theta, self.featureScaler, self.dataset)
        cfs_ord = org2ord(cfs_org, self.dataset)
        cfs_ord = pd.DataFrame(data=cfs_ord, columns=self.feature_names)
        cfs_ord.drop_duplicates(keep="first", inplace=True)

        # selecting the top n_cf counterfactuals
        cfs_ord = cfs_ord.iloc[:self.n_cf,:]

        # selecting the best counterfactual
        best_cf_ord = cfs_ord.iloc[0]

        ## returning the results
        explanations = {'cfs_ord': cfs_ord,
                        'best_cf_ord': best_cf_ord,
                        'toolbox': self.toolbox,
                        'featureScaler': self.featureScaler,
                        'objective_names': self.objective_names,
                        'objective_weights': self.objective_weights
                        }

        return explanations