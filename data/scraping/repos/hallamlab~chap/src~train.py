'''
This file is the main entry used to train, evaluate, and transform 
the dataset using various hierarchal Bayesian mixture models.
'''

import os
import sys
import time
import traceback

import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation as skLDA

from model.ctm import CorrelatedTopicModel as CTM
from model.soap import SparseCorrelatedBagPathway as SOAP
from model.spreat import diSparseCorrelatedBagPathway as SPREAT
from utility.access_file import save_data, load_data


###***************************        Private Main Entry        ***************************###


def __train(arg):
    # Setup the number of operations to employ
    steps = 1
    # Whether to display parameters at every operation
    display_params = True

    ##########################################################################################################
    ######################                            TRAIN                             ######################
    ##########################################################################################################

    if arg.train:
        print('\t>> Loading files...')
        dictionary = load_data(
            file_name=arg.vocab_name, load_path=arg.dspath, tag="dictionary", print_tag=False)
        X = load_data(file_name=arg.X_name, load_path=arg.dspath,
                      tag="X", print_tag=False)
        M = None
        features = None
        if arg.use_supplement:
            M = load_data(file_name=arg.M_name, load_path=arg.dspath,
                          tag="supplementary components")
            M = M.toarray()
        if arg.use_features:
            features = load_data(file_name=arg.features_name,
                                 load_path=arg.dspath, tag="features")

        if arg.soap:
            print('\n{0})- Training using SOAP model...'.format(steps))
            steps = steps + 1
            model_name = 'soap_' + arg.model_name
            model = SOAP(vocab=dictionary.token2id, num_components=arg.num_components,
                         alpha_mu=arg.alpha_mu, alpha_sigma=arg.alpha_sigma,
                         alpha_phi=arg.alpha_phi, gamma=arg.gamma, kappa=arg.kappa,
                         xi=arg.xi, varpi=arg.varpi, optimization_method=arg.opt_method,
                         cost_threshold=arg.cost_threshold, component_threshold=arg.component_threshold,
                         max_sampling=arg.max_sampling, subsample_input_size=arg.subsample_input_size,
                         batch=arg.batch, num_epochs=arg.num_epochs,
                         max_inner_iter=arg.max_inner_iter, top_k=arg.top_k, collapse2ctm=arg.collapse2ctm,
                         use_features=arg.use_features, num_jobs=arg.num_jobs,
                         display_interval=arg.display_interval, shuffle=arg.shuffle,
                         forgetting_rate=arg.forgetting_rate, delay_factor=arg.delay_factor,
                         random_state=arg.random_state, log_path=arg.logpath)
            model.fit(X=X, M=M, features=features, model_name=model_name, model_path=arg.mdpath,
                      result_path=arg.rspath, display_params=display_params)

        if arg.spreat:
            print('\n{0})- Training using SPREAT model...'.format(steps))
            steps = steps + 1
            model_name = 'spreat_' + arg.model_name
            model = SPREAT(vocab=dictionary.token2id, num_components=arg.num_components,
                           alpha_mu=arg.alpha_mu, alpha_sigma=arg.alpha_sigma,
                           alpha_phi=arg.alpha_phi, gamma=arg.gamma, kappa=arg.kappa,
                           xi=arg.xi, varpi=arg.varpi, optimization_method=arg.opt_method,
                           cost_threshold=arg.cost_threshold, component_threshold=arg.component_threshold,
                           max_sampling=arg.max_sampling, subsample_input_size=arg.subsample_input_size,
                           batch=arg.batch, num_epochs=arg.num_epochs,
                           max_inner_iter=arg.max_inner_iter, top_k=arg.top_k, collapse2ctm=arg.collapse2ctm,
                           use_features=arg.use_features, num_jobs=arg.num_jobs,
                           display_interval=arg.display_interval, shuffle=arg.shuffle,
                           forgetting_rate=arg.forgetting_rate, delay_factor=arg.delay_factor,
                           random_state=arg.random_state, log_path=arg.logpath)
            model.fit(X=X, M=M, features=features, model_name=model_name, model_path=arg.mdpath,
                      result_path=arg.rspath, display_params=display_params)

        if arg.ctm:
            print('\n{0})- Training using CMT model...'.format(steps))
            steps = steps + 1
            model_name = 'ctm_' + arg.model_name
            model = CTM(vocab=dictionary.token2id, num_components=arg.num_components, alpha_mu=arg.alpha_mu,
                        alpha_sigma=arg.alpha_sigma, alpha_beta=arg.alpha_phi,
                        optimization_method=arg.opt_method, cost_threshold=arg.cost_threshold,
                        component_threshold=arg.component_threshold, subsample_input_size=arg.subsample_input_size,
                        batch=arg.batch, num_epochs=arg.num_epochs, max_inner_iter=arg.max_inner_iter,
                        num_jobs=arg.num_jobs, display_interval=arg.display_interval,
                        shuffle=arg.shuffle, forgetting_rate=arg.forgetting_rate,
                        delay_factor=arg.delay_factor, random_state=arg.random_state,
                        log_path=arg.logpath)
            model.fit(X=X, model_name=model_name, model_path=arg.mdpath,
                      result_path=arg.rspath, display_params=display_params)

        if arg.lda:
            print('\n{0})- Training using LDA (sklearn) model...'.format(steps))
            steps = steps + 1
            model_name = 'sklda_' + arg.model_name
            model = skLDA(n_components=arg.num_components, learning_method='batch',
                          learning_decay=arg.delay_factor, learning_offset=arg.forgetting_rate,
                          max_iter=1, batch_size=arg.batch, evaluate_every=arg.display_interval,
                          perp_tol=arg.cost_threshold, mean_change_tol=arg.component_threshold,
                          max_doc_update_iter=arg.max_inner_iter, n_jobs=arg.num_jobs, verbose=0,
                          random_state=arg.random_state)
            print('\t>> Training by LDA model...')
            n_epochs = arg.num_epochs + 1
            old_bound = np.inf
            num_samples = int(X.shape[0] * arg.subsample_input_size)
            list_batches = np.arange(start=0, stop=num_samples, step=arg.batch)
            cost_file_name = model_name + "_cost.txt"
            save_data('', file_name=cost_file_name, save_path=arg.rspath,
                      mode='w', w_string=True, print_tag=False)
            for epoch in np.arange(start=1, stop=n_epochs):
                desc = '\t   {0:d})- Epoch count ({0:d}/{1:d})...'.format(
                    epoch, n_epochs - 1)
                print(desc)
                idx = np.random.choice(X.shape[0], num_samples, False)
                start_epoch = time.time()
                X_tmp = X[idx, :]
                for bidx, batch in enumerate(list_batches):
                    desc = '\t       --> Training: {0:.2f}%...'.format(
                        ((bidx + 1) / len(list_batches)) * 100)
                    if (bidx + 1) != len(list_batches):
                        print(desc, end="\r")
                    if (bidx + 1) == len(list_batches):
                        print(desc)
                    model.partial_fit(X=X_tmp[batch:batch + arg.batch])
                end_epoch = time.time()
                new_bound = - model.score(X=X_tmp) / X.shape[1]
                new_bound = np.log(new_bound)
                print('\t\t  ## Epoch {0} took {1} seconds...'.format(
                    epoch, round(end_epoch - start_epoch, 3)))
                data = str(epoch) + '\t' + str(round(end_epoch -
                                                     start_epoch, 3)) + '\t' + str(new_bound) + '\n'
                save_data(data=data, file_name=cost_file_name, save_path=arg.rspath, mode='a', w_string=True,
                          print_tag=False)
                print(
                    '\t\t  --> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_bound, old_bound))
                if new_bound <= old_bound or epoch == n_epochs - 1:
                    print(
                        '\t\t  --> Storing the LDA phi to: {0:s}'.format(model_name + '_phi.npz'))
                    np.savez(os.path.join(arg.mdpath, model_name +
                                          '_phi.npz'), model.components_)
                    print(
                        '\t\t  --> Storing the LDA (sklearn) model to: {0:s}'.format(model_name + '.pkl'))
                    save_data(data=model, file_name=model_name + '.pkl',
                              save_path=arg.mdpath, mode="wb", print_tag=False)
                    if epoch == n_epochs - 1:
                        print(
                            '\t\t  --> Storing the LDA phi to: {0:s}'.format(model_name + '_phi_final.npz'))
                        np.savez(os.path.join(arg.mdpath, model_name +
                                              '_phi_final.npz'), model.components_)
                        print(
                            '\t\t  --> Storing the LDA (sklearn) model to: {0:s}'.format(model_name + '_final.pkl'))
                        save_data(data=model, file_name=model_name + '_final.pkl',
                                  save_path=arg.mdpath, mode="wb", print_tag=False)
                    old_bound = new_bound
        display_params = False

    ##########################################################################################################
    ######################                           EVALUATE                           ######################
    ##########################################################################################################

    if arg.evaluate:
        print('\t>> Loading files...')
        dictionary = load_data(file_name=arg.vocab_name, load_path=arg.dspath, tag="vocabulary",
                               print_tag=False)
        X = load_data(file_name=arg.X_name, load_path=arg.dspath, tag="X",
                      print_tag=False)
        corpus = load_data(file_name=arg.text_name, load_path=arg.dspath, tag="X (a list of strings)",
                           print_tag=False)
        data = [[dictionary[i] for i, j in item] for item in corpus]

        M = None
        features = None
        if arg.use_supplement:
            M = load_data(file_name=arg.M_name, load_path=arg.dspath,
                          tag="supplementary components")
            M = M.toarray()
        if arg.use_features:
            features = load_data(file_name=arg.features_name,
                                 load_path=arg.dspath, tag="features")

        if arg.soap:
            print('\n{0})- Evaluating SOAP model...'.format(steps))
            steps = steps + 1
            model_name = 'soap_' + arg.model_name + '.pkl'
            file_name = 'soap_' + arg.model_name + '_score.txt'
            print('\t>> Loading SOAP model...')
            model = load_data(file_name=model_name, load_path=arg.mdpath, tag='SOAP model',
                              print_tag=False)
            score = model.predictive_distribution(X=X, M=M, features=features, cal_average=arg.cal_average,
                                                  batch_size=arg.batch, num_jobs=arg.num_jobs)
            print("\t>> Average log predictive score: {0:.4f}".format(score))
            save_data(data="# Average log predictive score: {0:.10f}\n".format(score),
                      file_name=file_name, save_path=arg.rspath, tag="log predictive score",
                      mode='w', w_string=True, print_tag=False)
            components = np.argsort(-model.phi)[:, :arg.top_k]
            components = [[dictionary[i] for i in item] for item in components]
            for cr in ['u_mass', 'c_v', 'c_uci', 'c_npmi']:
                cm = CoherenceModel(texts=data, topics=components,
                                    corpus=corpus, dictionary=dictionary, coherence=cr)
                coherence = cm.get_coherence()
                print("\t>> Average coherence ({0}) score: {1:.4f}".format(
                    cr, coherence))
                save_data(data="# Average coherence ({0}) score: {1:.4f}\n".format(cr, coherence),
                          file_name=file_name, save_path=arg.rspath, tag="coherence score",
                          mode='a', w_string=True, print_tag=False)

        if arg.spreat:
            print('\n{0})- Evaluating SPREAT model...'.format(steps))
            steps = steps + 1
            model_name = 'spreat_' + arg.model_name + '.pkl'
            file_name = 'spreat_' + arg.model_name + '_score.txt'
            print('\t>> Loading SPREAT model...')
            model = load_data(file_name=model_name, load_path=arg.mdpath, tag='SPREAT model',
                              print_tag=False)
            score = model.predictive_distribution(X=X, M=M, features=features, cal_average=arg.cal_average,
                                                  batch_size=arg.batch, num_jobs=arg.num_jobs)
            print("\t>> Average log predictive score: {0:.4f}".format(score))
            save_data(data="# Average log predictive score: {0:.10f}\n".format(score),
                      file_name=file_name, save_path=arg.rspath, tag="log predictive score",
                      mode='w', w_string=True, print_tag=False)
            components = np.argsort(-model.phi)[:, :arg.top_k]
            components = [[dictionary[i] for i in item] for item in components]
            for cr in ['u_mass', 'c_v', 'c_uci', 'c_npmi']:
                cm = CoherenceModel(texts=data, topics=components,
                                    corpus=corpus, dictionary=dictionary, coherence=cr)
                coherence = cm.get_coherence()
                print("\t>> Average coherence ({0}) score: {1:.4f}".format(
                    cr, coherence))
                save_data(data="# Average coherence ({0}) score: {1:.4f}\n".format(cr, coherence),
                          file_name=file_name, save_path=arg.rspath, tag="coherence score",
                          mode='a', w_string=True, print_tag=False)

        if arg.ctm:
            print('\n{0})- Evaluating CTM model...'.format(steps))
            steps = steps + 1
            model_name = 'ctm_' + arg.model_name + '.pkl'
            file_name = 'ctm_' + arg.model_name + '_score.txt'
            print('\t>> Loading CTM model...')
            model = load_data(file_name=model_name, load_path=arg.mdpath, tag='CTM model',
                              print_tag=False)
            score = model.predictive_distribution(
                X=X, cal_average=arg.cal_average, batch_size=arg.batch, num_jobs=arg.num_jobs)
            print("\t>> Average log predictive score: {0:.4f}".format(score))
            save_data(data="# Average log predictive score: {0:.10f}\n".format(score),
                      file_name=file_name, save_path=arg.rspath, tag="log predictive score",
                      mode='w', w_string=True, print_tag=False)
            components = np.argsort(-model.omega)[:, :arg.top_k]
            components = [[dictionary[i] for i in item] for item in components]
            for cr in ['u_mass', 'c_v', 'c_uci', 'c_npmi']:
                cm = CoherenceModel(texts=data, topics=components,
                                    corpus=corpus, dictionary=dictionary, coherence=cr)
                coherence = cm.get_coherence()
                print("\t>> Average coherence ({0}) score: {1:.4f}".format(
                    cr, coherence))
                save_data(data="# Average coherence ({0}) score: {1:.4f}\n".format(cr, coherence),
                          file_name=file_name, save_path=arg.rspath, tag="coherence score",
                          mode='a', w_string=True, print_tag=False)

        if arg.lda:
            print('\n{0})- Evaluating LDA model...'.format(steps))
            steps = steps + 1
            model_name = 'sklda_' + arg.model_name + '.pkl'
            file_name = 'sklda_' + arg.model_name + '_score.txt'
            print('\t>> Loading LDA model...')
            model = load_data(file_name=model_name, load_path=arg.mdpath, tag='LDA model',
                              print_tag=False)
            model.components_ /= model.components_.sum(1)[:, np.newaxis]
            component_distribution = model.transform(X=X)
            score = 0.0
            for idx in np.arange(X.shape[0]):
                feature_idx = X[idx].indices
                temp = np.multiply(
                    component_distribution[idx][:, np.newaxis], model.components_[:, feature_idx])
                score += np.sum(temp)
            if arg.cal_average:
                score = score / X.shape[0]
            score = np.log(score + np.finfo(np.float).eps)
            print("\t>> Average log predictive score: {0:.4f}".format(score))
            save_data(data="# Average log predictive score: {0:.10f}\n".format(score),
                      file_name=file_name, save_path=arg.rspath, tag="log predictive score",
                      mode='w', w_string=True, print_tag=False)
            components = np.argsort(-model.components_)[:, :arg.top_k]
            components = [[dictionary[i] for i in item] for item in components]
            for cr in ['u_mass', 'c_v', 'c_uci', 'c_npmi']:
                cm = CoherenceModel(texts=data, topics=components,
                                    corpus=corpus, dictionary=dictionary, coherence=cr)
                coherence = cm.get_coherence()
                print("\t>> Average coherence ({0}) score: {1:.4f}".format(
                    cr, coherence))
                save_data(data="# Average coherence ({0}) score: {1:.4f}\n".format(cr, coherence),
                          file_name=file_name, save_path=arg.rspath, tag="coherence score",
                          mode='a', w_string=True, print_tag=False)

    ##########################################################################################################
    ######################                           TRANSFORM                          ######################
    ##########################################################################################################

    if arg.transform:
        print('\t>> Loading files...')
        X = load_data(file_name=arg.X_name, load_path=arg.dspath,
                      tag="X", print_tag=False)

        M = None
        features = None
        if arg.use_supplement:
            M = load_data(file_name=arg.M_name, load_path=arg.dspath,
                          tag="supplementary components")
            M = M.toarray()
        if arg.use_features:
            features = load_data(file_name=arg.features_name,
                                 load_path=arg.dspath, tag="features")

        if arg.soap:
            print(
                '\n{0})- Transforming {1} using a pre-trained SOAP model...'.format(steps, arg.X_name))
            steps = steps + 1
            model_name = 'soap_' + arg.model_name + '.pkl'
            file_name = 'soap_' + arg.file_name + '.pkl'
            print('\t>> Loading SOAP model...')
            model = load_data(
                file_name=model_name, load_path=arg.mdpath, tag='SOAP model', print_tag=False)
            X = model.transform(X=X, M=M, features=features,
                                batch_size=arg.batch, num_jobs=arg.num_jobs)
            save_data(data=X, file_name=file_name, save_path=arg.dspath,
                      tag="transformed X", mode='wb', print_tag=True)

        if arg.spreat:
            print(
                '\n{0})- Transforming {1} using a pre-trained SPREAT model...'.format(steps, arg.X_name))
            steps = steps + 1
            model_name = 'spreat_' + arg.model_name + '.pkl'
            file_name = 'spreat_' + arg.file_name + '.pkl'
            print('\t>> Loading SPREAT model...')
            model = load_data(
                file_name=model_name, load_path=arg.mdpath, tag='SPREAT model', print_tag=False)
            X = model.transform(X=X, M=M, features=features,
                                batch_size=arg.batch, num_jobs=arg.num_jobs)
            save_data(data=X, file_name=file_name, save_path=arg.dspath,
                      tag="transformed X", mode='wb', print_tag=True)

        if arg.ctm:
            print(
                '\n{0})- Transforming {1} using a pre-trained CTM model...'.format(steps, arg.X_name))
            steps = steps + 1
            model_name = 'ctm_' + arg.model_name + '.pkl'
            file_name = 'ctm_' + arg.file_name + '.pkl'
            print('\t>> Loading CTM model...')
            model = load_data(
                file_name=model_name, load_path=arg.mdpath, tag='CTM model', print_tag=False)
            X = model.transform(X=X, batch_size=arg.batch,
                                num_jobs=arg.num_jobs)
            save_data(data=X, file_name=file_name, save_path=arg.dspath,
                      tag="transformed X", mode='wb', print_tag=True)

        if arg.lda:
            print(
                '\n{0})- Transforming {1} using a pre-trained LDA model...'.format(steps, arg.X_name))
            steps = steps + 1
            model_name = 'sklda_' + arg.model_name + '.pkl'
            file_name = 'sklda_' + arg.file_name + '.pkl'
            print('\t>> Loading LDA model...')
            model = load_data(
                file_name=model_name, load_path=arg.mdpath, tag='LDA model', print_tag=False)
            X = model.transform(X=X)
            save_data(data=X, file_name=file_name, save_path=arg.dspath,
                      tag="transformed X", mode='wb', print_tag=True)


def train(arg):
    try:
        if arg.train or arg.evaluate or arg.transform:
            actions = list()
            if arg.train:
                actions += ['TRAIN MODELs']
            if arg.evaluate:
                actions += ['EVALUATE MODELs']
            if arg.transform:
                actions += ['TRANSFORM RESULTS USING SPECIFIED MODELs']
            desc = [str(item[0] + 1) + '. ' + item[1]
                    for item in zip(list(range(len(actions))), actions)]
            desc = ' '.join(desc)
            print('\n*** APPLIED ACTIONS ARE: {0}'.format(desc))
            timeref = time.time()
            __train(arg)
            print('\n*** The selected actions consumed {1:f} SECONDS\n'.format('', round(time.time() - timeref, 3)),
                  file=sys.stderr)
        else:
            print('\n*** PLEASE SPECIFY AN ACTION...\n', file=sys.stderr)
    except Exception:
        print(traceback.print_exc())
        raise
