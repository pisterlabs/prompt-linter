"""
* *IntelComp H2020 project*
* *Topic Modeling Toolbox*

Provides several classes for Topic Modeling management, representation, and curation
    - TMManager: Management of topic models and domain classification models
    - TMmodel: Generic representation of all topic models that serve for its curation
"""

import argparse
import itertools
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rbo
import scipy.sparse as sparse
from sparse_dot_topn import awesome_cossim_topn


class TMManager(object):
    """
    Main class to manage functionality for the management of topic models
    """

    def listTMmodels(self, path_TMmodels: Path):
        """
        Returns a dictionary with all topic models or all DC models

        Parameters
        ----------
        path_TMmodels : pathlib.Path
            Path to the folder hosting the topic models or the dc models

        Returns
        -------
        allTMmodels : Dictionary (path -> dictionary)
            One dictionary entry per model
            key is the topic model name
            value is a dictionary with metadata
        """
        allTMmodels = {}
        modelFolders = [el for el in path_TMmodels.iterdir()]

        for TMf in modelFolders:
            # For topic models
            if TMf.joinpath('trainconfig.json').is_file():
                # print(f"{TMf.as_posix()} is a topic model")
                modelConfig = TMf.joinpath('trainconfig.json')
                with modelConfig.open('r', encoding='utf8') as fin:
                    modelInfo = json.load(fin)
                    allTMmodels[modelInfo['name']] = {
                        "name": modelInfo['name'],
                        "description": modelInfo['description'],
                        "visibility": modelInfo['visibility'],
                        "creator": modelInfo['creator'],
                        "trainer": modelInfo['trainer'],
                        "TrDtSet": modelInfo['TrDtSet'],
                        "creation_date": modelInfo['creation_date'],
                        "hierarchy-level": modelInfo['hierarchy-level'],
                        "htm-version": modelInfo['htm-version']
                    }
                    submodelFolders = [
                        el for el in TMf.iterdir()
                        if not el.as_posix().endswith("modelFiles")
                        and not el.as_posix().endswith("corpus.parquet")
                        and not el.as_posix().endswith("_old")]
                    for sub_TMf in submodelFolders:
                        submodelConfig = sub_TMf.joinpath('trainconfig.json')
                        if submodelConfig.is_file():
                            with submodelConfig.open('r', encoding='utf8') as fin:
                                submodelInfo = json.load(fin)
                                corpus = "Subcorpus created from " + \
                                    str(modelInfo['name'])
                                allTMmodels[submodelInfo['name']] = {
                                    "name": submodelInfo['name'],
                                    "description": submodelInfo['description'],
                                    "visibility": submodelInfo['visibility'],
                                    "creator": modelInfo['creator'],
                                    "trainer": submodelInfo['trainer'],
                                    "TrDtSet": corpus,
                                    "creation_date": submodelInfo['creation_date'],
                                    "hierarchy-level": submodelInfo['hierarchy-level'],
                                    "htm-version": submodelInfo['htm-version']
                                }
            # For DC models
            elif TMf.joinpath('dc_config.json').is_file():
                # print(f"{TMf.as_posix()} is a domain classifier model")
                modelConfig = TMf.joinpath('dc_config.json')
                with modelConfig.open('r', encoding='utf8') as fin:
                    modelInfo = json.load(fin)
                allTMmodels[modelInfo['name']] = {
                    "name": modelInfo['name'],
                    "description": modelInfo['description'],
                    "visibility": modelInfo['visibility'],
                    "creator": modelInfo['creator'],
                    "type": modelInfo['type'],
                    "corpus": modelInfo['corpus'],
                    "tag": modelInfo['tag'],
                    "creation_date": modelInfo['creation_date']
                }
            # This condition only applies for Mac OS
            elif TMf.name == ".DS_Store":
                pass
            else:
                print(f"No valid JSON file provided for Topic models or DC models")
                return 0
        return allTMmodels

    def getTMmodel(self, path_TMmodel: Path):
        """
        Returns a dictionary with a topic model and it's sub-models

        Parameters
        ----------
        path_TMmodel : pathlib.Path
            Path to the folder hosting the topic model

        Returns
        -------
        result : Dictionary (path -> dictionary)
            One dictionary entry per model
            key is the topic model name
            value is a dictionary with metadata
        """
        result = {}

        modelConfig = path_TMmodel.joinpath('trainconfig.json')
        if modelConfig.is_file():
            with modelConfig.open('r', encoding='utf8') as fin:
                modelInfo = json.load(fin)
                result[modelInfo['name']] = {
                    "name": modelInfo['name'],
                    "description": modelInfo['description'],
                    "visibility": modelInfo['visibility'],
                    "creator": modelInfo['creator'],
                    "trainer": modelInfo['trainer'],
                    "TMparam": modelInfo['TMparam'],
                    "creation_date": modelInfo['creation_date'],
                    "hierarchy-level": modelInfo['hierarchy-level'],
                    "htm-version": modelInfo['htm-version']
                }
            submodelFolders = [el for el in path_TMmodel.iterdir() if not el.as_posix().endswith(
                "modelFiles") and not el.as_posix().endswith("corpus.parquet") and not el.as_posix().endswith("_old")]
            for sub_TMf in submodelFolders:
                submodelConfig = sub_TMf.joinpath('trainconfig.json')
                if submodelConfig.is_file():
                    with submodelConfig.open('r', encoding='utf8') as fin:
                        submodelInfo = json.load(fin)
                        corpus = "Subcorpus created from " + \
                            str(modelInfo['name'])
                        result[submodelInfo['name']] = {
                            "name": submodelInfo['name'],
                            "description": submodelInfo['description'],
                            "visibility": submodelInfo['visibility'],
                            "creator": modelInfo['creator'],
                            "trainer": submodelInfo['trainer'],
                            "TrDtSet": corpus,
                            "TMparam": submodelInfo['TMparam'],
                            "creation_date": submodelInfo['creation_date'],
                            "hierarchy-level": submodelInfo['hierarchy-level'],
                            "htm-version": submodelInfo['htm-version']
                        }
        return result

    def deleteTMmodel(self, path_TMmodel: Path):
        """
        Deletes a Topic Model or a DC model

        Parameters
        ----------
        path_TMmodel : pathlib.Path
            Path to the folder containing the Topic Model or the DC model

        Returns
        -------
        status : int
            - 0 if the model could not be deleted
            - 1 if the model was deleted successfully
        """

        if not path_TMmodel.is_dir():
            print(f"File '{path_TMmodel.as_posix()}' does not exist.")
            return 0
        else:
            try:
                shutil.rmtree(path_TMmodel)
                return 1
            except:
                return 0

    def renameTMmodel(self, name: Path, new_name: Path):
        """
        Renames a topic model or a DC model

        Parameters
        ----------
        name : pathlib.Path
            Path to the model to be renamed

        new_name : pathlib.Path
            Path to the new name for the model

        Returns
        -------
        status : int
            - 0 if the model could not be renamed
            - 1 if the model was renamed successfully

        """
        if not name.is_dir():
            print(f"Model '{name.as_posix()}' does not exist.")
            return 0
        if new_name.is_file():
            print(
                f"Model '{new_name.as_posix()}' already exists. Rename or delete it first.")
            return 0
        try:
            # Checking whether it is a TM or DC model
            if name.joinpath('trainconfig.json').is_file():
                config_file = name.joinpath('trainconfig.json')
            elif name.joinpath('dc_config.json').is_file():
                config_file = name.joinpath('dc_config.json')
            with config_file.open("r", encoding="utf8") as fin:
                TMmodel = json.load(fin)
            TMmodel["name"] = new_name.stem
            with config_file.open("w", encoding="utf-8") as fout:
                json.dump(TMmodel, fout, ensure_ascii=False,
                          indent=2, default=str)
            shutil.move(name, new_name)
            return 1
        except:
            return 0

    def copyTMmodel(self, name: Path, new_name: Path):
        """
        Makes a copy of an existing TM or DC model

        Parameters
        ----------
        name : pathlib.Path
            Path to the model to be copied

        new_name : pathlib.Path
            Path to the new name for the model

        Returns
        -------
        status : int
            - 0 if the model could not be copied
            - 1 if the model was copied successfully

        """
        if not name.is_dir():
            print(f"Model '{name.as_posix()}' does not exist.")
            return 0
        if new_name.is_file():
            print(
                f"Model '{new_name.as_posix()}' already exists. Rename or delete it first.")
            return 0
        try:
            shutil.copytree(name, new_name)

            # Checking whether it is a TM or DC model
            if new_name.joinpath('trainconfig.json').is_file():
                config_file = name.joinpath('trainconfig.json')
            elif new_name.joinpath('dc_config.json').is_file():
                config_file = name.joinpath('dc_config.json')
            with config_file.open("r", encoding="utf8") as fin:
                TMmodel = json.load(fin)
            TMmodel["name"] = new_name.stem
            with config_file.open("w", encoding="utf-8") as fout:
                json.dump(TMmodel, fout, ensure_ascii=False,
                          indent=2, default=str)
            return 1
        except:
            return 0


class TMmodel(object):
    # This class represents a Topic Model according to the LDA generative model
    # Essentially the TM is characterized by
    # _alphas: The weight of each topic
    # _betas: The weight of each word in the vocabulary
    # _thetas: The weight of each topic in each document
    #
    # and needs to be backed up with a folder in which all the associated
    # files will be saved
    #
    # The TM can be trained with Blei's LDA, Mallet, or any other toolbox
    # that produces a model according to this representation

    # The following variables will store original values of matrices alphas, betas, thetas
    # They will be used to reset the model to original values

    _TMfolder = None

    _betas_orig = None
    _thetas_orig = None
    _alphas_orig = None

    _betas = None
    _thetas = None
    _alphas = None
    _edits = None  # Store all editions made to the model
    _ntopics = None
    _betas_ds = None
    _coords = None
    _topic_entropy = None
    _topic_coherence = None
    _ndocs_active = None
    _tpc_descriptions = None
    _tpc_labels = None
    _vocab_w2id = None
    _vocab_id2w = None
    _vocab = None
    _size_vocab = None
    _sims = None

    def __init__(self, TMfolder, get_sims=False, logger=None):
        """Class initializer

        We just need to make sure that we have a folder where the
        model will be stored. If the folder does not exist, it will
        create a folder for the model

        Parameters
        ----------
        TMfolder: Path
            Contains the name of an existing folder or a new folder
            where the model will be created
        get_sims: boolean
            Flag to detect if similarities are going to be calculated or not.
        logger:
            External logger to use. If None, a logger will be created for the object
        """
        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('TMmodel')

        # Convert strings to Paths if necessary
        self._TMfolder = Path(TMfolder)

        # If folder already exists no further action is needed
        # in other case, the folder is created
        if not self._TMfolder.is_dir():
            try:
                self._TMfolder.mkdir(parents=True)
            except:
                self._logger.error(
                    '-- -- Topic model object (TMmodel) could not be created')

        self._logger.info(
            '-- -- -- Topic model object (TMmodel) successfully created')

        self._get_sims = get_sims

    def create(self, betas=None, thetas=None, alphas=None, vocab=None, labels=None):
        """Creates the topic model from the relevant matrices that characterize it. In addition to the initialization of the corresponding object's variables, all the associated variables and visualizations which are computationally costly are calculated so they are available for the other methods.

        Parameters
        ----------
        betas:
            Matrix of size n_topics x n_words (vocab of each topic)
        thetas:
            Matrix of size  n_docs x n_topics (document composition)
        alphas: 
            Vector of length n_topics containing the importance of each topic
        vocab: list
            List of words sorted according to betas matrix
        labels: list
            List of  labels for automatic topic labeling
        """

        # If folder already exists no further action is needed
        # in other case, the folder is created
        if not self._TMfolder.is_dir():
            self._logger.error(
                '-- -- Topic model object (TMmodel) folder not ready')
            return

        self._alphas_orig = alphas
        self._betas_orig = betas
        self._thetas_orig = thetas
        self._alphas = alphas
        self._betas = betas
        self._thetas = thetas
        self._vocab = vocab
        self._size_vocab = len(vocab)
        self._ntopics = thetas.shape[1]
        self._edits = []

        # Save original variables
        np.save(self._TMfolder.joinpath('alphas_orig.npy'), alphas)
        np.save(self._TMfolder.joinpath('betas_orig.npy'), betas)
        sparse.save_npz(self._TMfolder.joinpath('thetas_orig.npz'), thetas)
        with self._TMfolder.joinpath('vocab.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(vocab))

        # Initial sort of topics according to size. Calculate other variables
        self._sort_topics()
        self._calculate_beta_ds()
        self._calculate_topic_entropy()
        self._ndocs_active = np.array((self._thetas != 0).sum(0).tolist()[0])
        self._tpc_descriptions = [el[1]
                                  for el in self.get_tpc_word_descriptions()]
        self.calculate_topic_coherence()  # cohrs_aux
        self._tpc_labels = [el[1] for el in self.get_tpc_labels(labels)]
        if self._get_sims:
            self._calculate_sims()

        # We are ready to save all variables in the model
        self._save_all()

        self._logger.info(
            '-- -- Topic model variables were computed and saved to file')
        return

    def _save_all(self):
        """Saves all variables in Topic Model
        * alphas, betas, thetas
        * edits
        * betas_ds, topic_entropy, ndocs_active
        * tpc_descriptions, tpc_labels
        This function should only be called after making sure all these
        variables exist and are not None
        """
        np.save(self._TMfolder.joinpath('alphas.npy'), self._alphas)
        np.save(self._TMfolder.joinpath('betas.npy'), self._betas)
        sparse.save_npz(self._TMfolder.joinpath('thetas.npz'), self._thetas)
        if self._get_sims:
            sparse.save_npz(self._TMfolder.joinpath(
                'distances.npz'), self._sims)

        with self._TMfolder.joinpath('edits.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._edits))
        np.save(self._TMfolder.joinpath('betas_ds.npy'), self._betas_ds)
        np.save(self._TMfolder.joinpath(
            'topic_entropy.npy'), self._topic_entropy)
        np.save(self._TMfolder.joinpath(
            'topic_coherence.npy'), self._topic_coherence)
        np.save(self._TMfolder.joinpath(
            'ndocs_active.npy'), self._ndocs_active)
        with self._TMfolder.joinpath('tpc_descriptions.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._tpc_descriptions))
        with self._TMfolder.joinpath('tpc_labels.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._tpc_labels))

        # Generate also pyLDAvisualization
        # pyLDAvis currently raises some Deprecation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import pyLDAvis

        # We will compute the visualization using ndocs random documents
        # In case the model has gone through topic deletion, we may have rows
        # in the thetas matrix that sum up to zero (active topics have been
        # removed for these problematic documents). We need to take this into
        # account
        ndocs = 10000
        validDocs = np.sum(self._thetas.toarray(), axis=1) > 0
        nValidDocs = np.sum(validDocs)
        if ndocs > nValidDocs:
            ndocs = nValidDocs
        perm = np.sort(np.random.permutation(nValidDocs)[:ndocs])
        # We consider all documents are equally important
        doc_len = ndocs * [1]
        vocabfreq = np.round(ndocs*(self._alphas.dot(self._betas))).astype(int)
        vis_data = pyLDAvis.prepare(
            self._betas,
            self._thetas[validDocs, ][perm, ].toarray(),
            doc_len,
            self._vocab,
            vocabfreq,
            lambda_step=0.05,
            sort_topics=False,
            n_jobs=-1)

        # Save html
        with self._TMfolder.joinpath("pyLDAvis.html").open("w") as f:
            pyLDAvis.save_html(vis_data, f)
        # TODO: Check substituting by "pyLDAvis.prepared_data_to_html"
        # self._modify_pyldavis_html(self._TMfolder.as_posix())

        # Get coordinates of topics in the pyLDAvis visualization
        vis_data_dict = vis_data.to_dict()
        self._coords = list(
            zip(*[vis_data_dict['mdsDat']['x'], vis_data_dict['mdsDat']['y']]))

        with self._TMfolder.joinpath('tpc_coords.txt').open('w', encoding='utf8') as fout:
            for item in self._coords:
                fout.write(str(item) + "\n")

        return

    def _save_cohr(self):

        np.save(self._TMfolder.joinpath(
            'topic_coherence.npy'), self._topic_coherence)

    def _modify_pyldavis_html(self, model_dir):
        """
        Modifies the PyLDAvis HTML file returned by the Gensim library to include the direct paths of the 'd3.js' and 'ldavis.v3.0.0.js', which are copied into the model/submodel directory.

        Parameters
        ----------
        model_dir: str
            String representation of the path wwhere the model/submodel is located
        """

        # Copy necessary files in model / submodel folder for PyLDAvis visualization
        d3 = Path("src/gui/resources/d3.js")
        v3 = Path("src/gui/resources/ldavis.v3.0.0.js")
        shutil.copyfile(d3, Path(model_dir, "d3.js"))
        shutil.copyfile(v3, Path(model_dir, "ldavis.v3.0.0.js"))

        # Update d3 and v3 paths in pyldavis.html
        fin = open(Path(model_dir, "pyLDAvis.html").as_posix(),
                   "rt")  # read input file
        data = fin.read()  # read file contents to string
        # Replace all occurrences of the required string
        data = data.replace(
            "https://d3js.org/d3.v5.js", "d3.js")
        data = data.replace(
            "https://d3js.org/d3.v5", "d3.js")
        data = data.replace(
            "https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.3.1/pyLDAvis/js/ldavis.v3.0.0.js", "ldavis.v3.0.0.js")
        fin.close()  # close the input file
        fin = open(Path(model_dir, "pyLDAvis.html").as_posix(),
                   "wt")  # open the input file in write mode
        fin.write(data)  # overrite the input file with the resulting data
        fin.close()  # close the file

        return

    def _sort_topics(self):
        """Sort topics according to topic size"""

        # Load information if necessary
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_edits()

        # Indexes for topics reordering
        idx = np.argsort(self._alphas)[::-1]
        self._edits.append('s ' + ' '.join([str(el) for el in idx]))

        # Sort data matrices
        self._alphas = self._alphas[idx]
        self._betas = self._betas[idx, :]
        self._thetas = self._thetas[:, idx]
        
        # Save equivalences mapping between original Topic IDs and new IDs
        corr = {int(idx[i]): i for i in range(len(self._alphas))}
        with open(self._TMfolder.joinpath('corr.json').as_posix(), 'w') as json_file:
            json.dump(corr, json_file)

        return

    def _load_alphas(self):
        if self._alphas is None:
            self._alphas = np.load(self._TMfolder.joinpath('alphas.npy'))
            self._ntopics = self._alphas.shape[0]

    def _load_betas(self):
        if self._betas is None:
            self._betas = np.load(self._TMfolder.joinpath('betas.npy'))
            self._ntopics = self._betas.shape[0]
            self._size_vocab = self._betas.shape[1]

    def _load_thetas(self):
        if self._thetas is None:
            self._thetas = sparse.load_npz(
                self._TMfolder.joinpath('thetas.npz'))
            self._ntopics = self._thetas.shape[1]
            # self._ndocs_active = np.array((self._thetas != 0).sum(0).tolist()[0])

    def _load_ndocs_active(self):
        if self._ndocs_active is None:
            self._ndocs_active = np.load(
                self._TMfolder.joinpath('ndocs_active.npy'))
            self._ntopics = self._ndocs_active.shape[0]

    def _load_edits(self):
        if self._edits is None:
            with self._TMfolder.joinpath('edits.txt').open('r', encoding='utf8') as fin:
                self._edits = fin.readlines()

    def _calculate_beta_ds(self):
        """Calculates beta with down-scoring
        Emphasizes words appearing less frequently in topics
        """
        # Load information if necessary
        self._load_betas()

        self._betas_ds = np.copy(self._betas)
        if np.min(self._betas_ds) < 1e-12:
            self._betas_ds += 1e-12
        deno = np.reshape((sum(np.log(self._betas_ds)) /
                          self._ntopics), (self._size_vocab, 1))
        deno = np.ones((self._ntopics, 1)).dot(deno.T)
        self._betas_ds = self._betas_ds * (np.log(self._betas_ds) - deno)

    def _load_betas_ds(self):
        if self._betas_ds is None:
            self._betas_ds = np.load(self._TMfolder.joinpath('betas_ds.npy'))
            self._ntopics = self._betas_ds.shape[0]
            self._size_vocab = self._betas_ds.shape[1]

    def _load_vocab(self):
        if self._vocab is None:
            with self._TMfolder.joinpath('vocab.txt').open('r', encoding='utf8') as fin:
                self._vocab = [el.strip() for el in fin.readlines()]

    def _load_vocab_dicts(self):
        """Creates two vocabulary dictionaries, one that utilizes the words as key, and a second one with the words' id as key. 
        """
        if self._vocab_w2id is None and self._vocab_w2id is None:
            self._vocab_w2id = {}
            self._vocab_id2w = {}
            with self._TMfolder.joinpath('vocab.txt').open('r', encoding='utf8') as fin:
                for i, line in enumerate(fin):
                    wd = line.strip()
                    self._vocab_w2id[wd] = i
                    self._vocab_id2w[str(i)] = wd

    def _calculate_topic_entropy(self):
        """Calculates the entropy of all topics in model
        """
        # Load information if necessary
        self._load_betas()

        if np.min(self._betas) < 1e-12:
            self._betas += 1e-12
        self._topic_entropy = - \
            np.sum(self._betas * np.log(self._betas), axis=1)
        self._topic_entropy = self._topic_entropy / np.log(self._size_vocab)

    def _load_topic_entropy(self):
        if self._topic_entropy is None:
            self._topic_entropy = np.load(
                self._TMfolder.joinpath('topic_entropy.npy'))

    def calculate_topic_coherence(self,
                                  metrics=["c_v", "c_npmi"],
                                  n_words=15,
                                  reference_text=None,
                                  only_one=True,
                                  aggregated=False) -> list:
        """Calculates the per-topic coherence of a topic model, given as TMmodel, or its average coherence when aggregated is True.

        If only_one is False and metrics is a list of different coherence metrics, the function returns a list of lists, where each sublist contains the coherence values for the respective metric.

        If reference_text is given, the coherence is calculated with respect to this text. Otherwise, the coherence is calculated with respect to the corpus used to train the topic model.

        Parameters
        ----------
        metrics : list of str, optional
            List of coherence metrics to be calculated. Possible values are 'c_v', 'c_uci', 'c_npmi', 'u_mass'. 
            The default is ["c_v", "c_npmi"].
        n_words : int, optional
            Number of words to be used for calculating the coherence. The default is 15.
        reference_text : str, optional
            Text to be used as reference for calculating the coherence. The default is None.
        only_one : bool, optional
            If True, only one coherence value is returned. If False, a list of coherence values is returned. The default is True.
        aggregated : bool, optional
            If True, the average coherence of the topic model is returned. If False, the coherence of each topic is returned. The default is False.
        """

        # Load topic information
        if self._tpc_descriptions is None:
            self._tpc_descriptions = \
                [el[1] for el in self.get_tpc_word_descriptions()]

        # Convert topic information into list of lists (Gensim's Coherence Model format)
        tpc_descriptions_ = \
            [tpc.split(', ') for tpc in self._tpc_descriptions]

        if reference_text is None:
            # Get text to calculate coherence
            if self._TMfolder.parent.joinpath('modelFiles/corpus.txt').is_file():
                corpusFile = self._TMfolder.parent.joinpath(
                    'modelFiles/corpus.txt')
            else:
                corpusFile = self._TMfolder.parent.joinpath('corpus.txt')
            with corpusFile.open("r", encoding="utf-8") as f:
                corpus = [line.rsplit(" 0 ")[1].strip().split() for line in f.readlines(
                ) if line.rsplit(" 0 ")[1].strip().split() != []]
        else:
            # Texts should be given as a list of lists of strings
            corpus = reference_text

        # Import necessary modules for coherence calculation with Gensim
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel

        # Get Gensim dictionary
        dictionary = None
        if self._TMfolder.parent.joinpath('dictionary.gensim').is_file():
            try:
                dictionary = Dictionary.load_from_text(
                    self._TMfolder.parent.joinpath('dictionary.gensim').as_posix())
            except:
                self._logger.warning(
                    "Gensim dictionary could not be load from vocabulary file.")
        else:
            if dictionary is None:
                dictionary = Dictionary(corpus)

        if n_words > len(tpc_descriptions_[0]):
            self._logger.error(
                '-- -- -- Coherence calculation failed: The number of words per topic must be equal to n_words.')
            return None
        else:
            if only_one:
                metric = metrics[0]
                self._logger.info(
                    f"Calculating just coherence {metric}.")
                if metric in ["c_npmi", "u_mass", "c_v", "c_uci"]:
                    cm = CoherenceModel(topics=tpc_descriptions_, texts=corpus,
                                        dictionary=dictionary, coherence=metric, topn=n_words)
                    self._topic_coherence = cm.get_coherence_per_topic()

                    if aggregated:
                        mean = cm.aggregate_measures(self._topic_coherence)
                        return mean
                    return self._topic_coherence
                else:
                    self._logger.error(
                        '-- -- -- Coherence metric provided is not available.')
                    return None
            else:
                cohrs_aux = []
                for metric in metrics:
                    self._logger.info(
                        f"Calculating coherence {metric}.")
                    if metric in ["c_npmi", "u_mass", "c_v", "c_uci"]:
                        cm = CoherenceModel(topics=tpc_descriptions_, texts=corpus,
                                            dictionary=dictionary, coherence=metric, topn=n_words)
                        aux = cm.get_coherence_per_topic()
                        cohrs_aux.extend(aux)
                        self._logger.info(cohrs_aux)
                    else:
                        self._logger.error(
                            '-- -- -- Coherence metric provided is not available.')
                        return None
                self._topic_coherence = cohrs_aux

        return self._topic_coherence

    def calculate_rbo(self,
                      weight: float = 1.0,
                      n_words: int = 15) -> float:
        """Calculates the rank_biased_overlap over the topics in a topic model.

        Parameters
        ----------
        weigth : float, optional
            Weight of each agreement at depth d: p**(d-1). When set to 1.0, there is no weight, the rbo returns to average overlap. The defau>
        n_words : int, optional
            Number of words to be used for calculating the rbo. The default is 15.

        Returns
        -------
        rbo : float
            Rank_biased_overlap
        """

        # Load topic information
        if self._tpc_descriptions is None:
            self._tpc_descriptions = \
                [el[1] for el in self.get_tpc_word_descriptions(n_words)]

        collect = []
        for list1, list2 in itertools.combinations(self._tpc_descriptions, 2):
            rbo_val = rbo.RankingSimilarity(
                list1.split(", "), list2.split(", ")).rbo(p=weight)
            collect.append(rbo_val)

        return 1 - np.mean(collect)

    def calculate_topic_diversity(self,
                                  n_words: int = 15) -> float:
        """Calculates the percentage of unique words in the topn words of all topics. Diversity close to 0 indicates redundant topics; diversity close to 1 indicates more varied topics.

        Parameters
        ----------
        n_words : int, optional
            Number of words to be used for calculating the rbo. The default is 15.

        Returns
        -------
        td : float
            Topic diversity
        """

        # Load topic information
        if self._tpc_descriptions is None:
            self._tpc_descriptions = \
                [el[1] for el in self.get_tpc_word_descriptions(n_words)]

        unique_words = set()
        for topic in self._tpc_descriptions:
            unique_words = unique_words.union(set(topic.split(", ")))
        td = len(unique_words) / (n_words * len(self._tpc_descriptions))
        return td

    def _load_topic_coherence(self):
        if self._topic_coherence is None:
            self._topic_coherence = np.load(
                self._TMfolder.joinpath('topic_coherence.npy'))

    def _calculate_sims(self, topn=50, lb=0):
        if self._thetas is None:
            self._load_thetas()
        thetas_sqrt = np.sqrt(self._thetas)
        thetas_col = thetas_sqrt.T
        self._sims = awesome_cossim_topn(thetas_sqrt, thetas_col, topn, lb)

    def _load_sims(self):
        if self._sims is None:
            self._sims = sparse.load_npz(
                self._TMfolder.joinpath('distances.npz'))

    def _largest_indices(self, ary, n):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        idx0, idx1 = np.unravel_index(indices, ary.shape)
        idx0 = idx0.tolist()
        idx1 = idx1.tolist()
        selected_idx = []
        for id0, id1 in zip(idx0, idx1):
            if id0 < id1:
                selected_idx.append((id0, id1, ary[id0, id1]))
        return selected_idx

    def get_model_info_for_hierarchical(self):
        """Returns the objects necessary for the creation of a level-2 topic model.
        """
        self._load_betas()
        self._load_thetas()
        self._load_vocab_dicts()

        return self._betas, self._thetas, self._vocab_w2id, self._vocab_id2w

    def get_model_info_for_vis(self):
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_vocab()
        if self._get_sims:
            self._load_sims()
        self.load_tpc_coords()

        return self._alphas, self._betas, self._thetas, self._vocab, self._sims, self._coords

    def get_tpc_word_descriptions(self, n_words=15, tfidf=True, tpc=None):
        """returns the chemical description of topics

        Parameters
        ----------
        n_words:
            Number of terms for each topic that will be included
        tfidf:
            If true, downscale the importance of words that appear
            in several topics, according to beta_ds (Blei and Lafferty, 2009)
        tpc:
            Topics for which the descriptions will be computed, e.g.: tpc = [0,3,4]
            If None, it will compute the descriptions for all topics  

        Returns
        -------
        tpc_descs: list of tuples
            Each element is a a term (topic_id, "word0, word1, ...")                      
        """

        # Load betas (including n_topics) and vocabulary
        if tfidf:
            self._load_betas_ds()
        else:
            self._load_betas()
        self._load_vocab()

        if not tpc:
            tpc = range(self._ntopics)

        tpc_descs = []
        for i in tpc:
            if tfidf:
                words = [self._vocab[idx2]
                         for idx2 in np.argsort(self._betas_ds[i])[::-1][0:n_words]]
            else:
                words = [self._vocab[idx2]
                         for idx2 in np.argsort(self._betas[i])[::-1][0:n_words]]
            tpc_descs.append((i, ', '.join(words)))
        return tpc_descs

    def load_tpc_descriptions(self):
        if self._tpc_descriptions is None:
            with self._TMfolder.joinpath('tpc_descriptions.txt').open('r', encoding='utf8') as fin:
                self._tpc_descriptions = [el.strip() for el in fin.readlines()]

    def get_tpc_labels(self, labels=None, use_cuda=True):
        """returns the labels of the topics in the model

        Parameters
        ----------
        labels: list
            List of labels for automatic topic labeling
        use_cuda: bool
            If True, use cuda.

        Returns
        -------
        tpc_labels: list of tuples
            Each element is a a term (topic_id, "label for topic topic_id")                    
        """
        if not labels:
            return [(i, "NA") for i, p in enumerate(self._tpc_descriptions)]

        if use_cuda:
            import torch
            if torch.cuda.is_available():
                device = 0
                self._logger.info("-- -- CUDA available: GPU will be used")
            else:
                device = -1
                self._logger.warning(
                    "-- -- 'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set 'use_cuda=False'"
                )
                self._logger.info(
                    "-- -- CUDA unavailable: GPU will not be used")
        else:
            device = -1
            self._logger.info("-- -- CUDA unavailable: GPU will not be used")

        from transformers import pipeline
        classifier = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli",
                              device=device)
        predictions = classifier(self._tpc_descriptions, labels)
        predictions = [(i, p["labels"][0]) for i, p in enumerate(predictions)]
        return predictions

    def load_tpc_labels(self):
        if self._tpc_labels is None:
            with self._TMfolder.joinpath('tpc_labels.txt').open('r', encoding='utf8') as fin:
                self._tpc_labels = [el.strip() for el in fin.readlines()]

    def load_tpc_coords(self):
        if self._coords is None:
            with self._TMfolder.joinpath('tpc_coords.txt').open('r', encoding='utf8') as fin:
                # read the data from the file and convert it back to a list of tuples
                self._coords = \
                    [tuple(map(float, line.strip()[1:-1].split(', ')))
                        for line in fin]

    def get_alphas(self):
        self._load_alphas()
        return self._alphas

    def showTopics(self):
        self._load_alphas()
        self._load_ndocs_active()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        TpcsInfo = [{"Size": str(round(el[0], 4)), "Label": el[1].strip(), "Word Description": el[2].strip(), "Ndocs Active": str(el[3])} for el in zip(
            self._alphas, self._tpc_labels, self._tpc_descriptions, self._ndocs_active)]

        return TpcsInfo

    def showTopicsAdvanced(self):
        self._load_alphas()
        self._load_ndocs_active()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_topic_entropy()
        self._load_topic_coherence()
        TpcsInfo = [{"Size": str(round(el[0], 4)), "Label": el[1].strip(), "Word Description": el[2].strip(), "Ndocs Active": str(el[3]), "Topics entropy": str(round(
            el[4], 4)), "Topics coherence": str(round(el[5], 4))} for el in zip(self._alphas, self._tpc_labels, self._tpc_descriptions, self._ndocs_active, self._topic_entropy, self._topic_coherence)]

        return TpcsInfo

    def setTpcLabels(self, TpcLabels):
        self._tpc_labels = [el.strip() for el in TpcLabels]
        self._load_alphas()
        # Check that the number of labels is consistent with model
        if len(TpcLabels) == self._ntopics:
            with self._TMfolder.joinpath('tpc_labels.txt').open('w', encoding='utf8') as fout:
                fout.write('\n'.join(self._tpc_labels))
            return 1
        else:
            return 0

    def deleteTopics(self, tpcs):
        """This is a costly operation, almost everything
        needs to get modified"""
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_betas_ds()
        self._load_topic_entropy()
        self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_ndocs_active()
        self._load_edits()
        self._load_vocab()

        try:
            # Get a list of the topics that should be kept
            tpc_keep = [k for k in range(self._ntopics) if k not in tpcs]
            tpc_keep = [k for k in tpc_keep if k < self._ntopics]

            # Calculate new variables
            self._thetas = self._thetas[:, tpc_keep]
            from sklearn.preprocessing import normalize
            self._thetas = normalize(self._thetas, axis=1, norm='l1')
            self._alphas = np.asarray(np.mean(self._thetas, axis=0)).ravel()
            self._ntopics = self._thetas.shape[1]
            self._betas = self._betas[tpc_keep, :]
            self._betas_ds = self._betas_ds[tpc_keep, :]
            self._ndocs_active = self._ndocs_active[tpc_keep]
            self._topic_entropy = self._topic_entropy[tpc_keep]
            self._topic_coherence = self._topic_coherence[tpc_keep]
            self._tpc_labels = [self._tpc_labels[i] for i in tpc_keep]
            self._tpc_descriptions = [
                self._tpc_descriptions[i] for i in tpc_keep]
            self._edits.append('d ' + ' '.join([str(k) for k in tpcs]))

            # We are ready to save all variables in the model
            self._save_all()

            self._logger.info(
                '-- -- Topics deletion successful. All variables saved to file')
            return 1
        except:
            self._logger.info(
                '-- -- Topics deletion generated an error. Operation failed')
            return 0

    def getSimilarTopics(self, npairs, thr=1e-3):
        """Obtains pairs of similar topics
        npairs: number of pairs of words
        thr: threshold for vocabulary thresholding
        """

        self._load_thetas()
        self._load_betas()

        # Part 1 - Coocurring topics
        # Highly correlated topics co-occure together
        # Topic mean
        med = np.asarray(np.mean(self._thetas, axis=0)).ravel()
        # Topic square mean
        thetas2 = self._thetas.multiply(self._thetas)
        med2 = np.asarray(np.mean(thetas2, axis=0)).ravel()
        # Topic stds
        stds = np.sqrt(med2 - med ** 2)
        # Topic correlation
        num = self._thetas.T.dot(
            self._thetas).toarray() / self._thetas.shape[0]
        num = num - med[..., np.newaxis].dot(med[np.newaxis, ...])
        deno = stds[..., np.newaxis].dot(stds[np.newaxis, ...])
        corrcoef = num / deno
        selected_coocur = self._largest_indices(
            corrcoef, self._ntopics + 2 * npairs)
        selected_coocur = [(el[0], el[1], el[2].astype(float))
                           for el in selected_coocur]

        # Part 2 - Topics with similar word composition
        # Computes inter-topic distance based on word distributions
        # using scipy implementation of Jensen Shannon distance
        from scipy.spatial.distance import jensenshannon

        # For a more efficient computation with very large vocabularies
        # we implement a threshold for restricting the distance calculation
        # to columns where any element is greater than threshold thr
        betas_aux = self._betas[:, np.where(self._betas.max(axis=0) > thr)[0]]
        js_mat = np.zeros((self._ntopics, self._ntopics))
        for k in range(self._ntopics):
            for kk in range(self._ntopics):
                js_mat[k, kk] = jensenshannon(
                    betas_aux[k, :], betas_aux[kk, :])
        JSsim = 1 - js_mat
        selected_worddesc = self._largest_indices(
            JSsim, self._ntopics + 2 * npairs)
        selected_worddesc = [(el[0], el[1], el[2].astype(float))
                             for el in selected_worddesc]

        similarTopics = {
            'Coocurring': selected_coocur,
            'Worddesc': selected_worddesc
        }

        return similarTopics

    def fuseTopics(self, tpcs):
        """This is a costly operation, almost everything
        needs to get modified"""
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        # self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_edits()
        self._load_vocab()

        try:
            # List of topics that will be merged
            tpcs = sorted(tpcs)

            # Calculate new variables
            # For beta we keep a weighted average of topic vectors
            weights = self._alphas[tpcs]
            bet = weights[np.newaxis, ...].dot(
                self._betas[tpcs, :]) / (sum(weights))
            # keep new topic vector in upper position and delete the others
            self._betas[tpcs[0], :] = bet
            self._betas = np.delete(self._betas, tpcs[1:], 0)
            # For theta we need to keep the sum. Since adding implies changing
            # structure, we need to convert to full matrix first
            # No need to renormalize
            thetas_full = self._thetas.toarray()
            thet = np.sum(thetas_full[:, tpcs], axis=1)
            thetas_full[:, tpcs[0]] = thet
            thetas_full = np.delete(thetas_full, tpcs[1:], 1)
            self._thetas = sparse.csr_matrix(thetas_full, copy=True)
            # Compute new alphas and number of topics
            self._alphas = np.asarray(np.mean(self._thetas, axis=0)).ravel()
            self._ntopics = self._thetas.shape[1]
            # Compute all other variables
            self._calculate_beta_ds()
            self._calculate_topic_entropy()
            self._ndocs_active = np.array(
                (self._thetas != 0).sum(0).tolist()[0])

            # Keep label and description of most significant topic
            for tpc in tpcs[1:][::-1]:
                del self._tpc_descriptions[tpc]
            # Recalculate chemical description of most significant topic
            self._tpc_descriptions[tpcs[0]] = self.get_tpc_word_descriptions(tpc=[tpcs[0]])[
                0][1]
            for tpc in tpcs[1:][::-1]:
                del self._tpc_labels[tpc]

            self.calculate_topic_coherence()
            self._edits.append('f ' + ' '.join([str(el) for el in tpcs]))
            # We are ready to save all variables in the model
            if self._get_sims:
                self._calculate_sims()
            self._save_all()

            self._logger.info(
                '-- -- Topics merging successful. All variables saved to file')
            return 1
        except:
            self._logger.info(
                '-- -- Topics merging generated an error. Operation failed')
            return 0

    def sortTopics(self):
        """This is a costly operation, almost everything
        needs to get modified"""
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_betas_ds()
        self._load_topic_entropy()
        self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_ndocs_active()
        self._load_edits()
        self._load_vocab()

        try:
            # Calculate order for the topics
            idx = np.argsort(self._alphas)[::-1]
            self._edits.append('s ' + ' '.join([str(el) for el in idx]))

            # Calculate new variables
            self._thetas = self._thetas[:, idx]
            self._alphas = self._alphas[idx]
            self._betas = self._betas[idx, :]
            self._betas_ds = self._betas_ds[idx, :]
            self._ndocs_active = self._ndocs_active[idx]
            self._topic_entropy = self._topic_entropy[idx]
            self._topic_coherence = self._topic_coherence[idx]
            self._tpc_labels = [self._tpc_labels[i] for i in idx]
            self._tpc_descriptions = [self._tpc_descriptions[i] for i in idx]
            self._edits.append('s ' + ' '.join([str(el) for el in idx]))

            # We are ready to save all variables in the model
            self._save_all()

            self._logger.info(
                '-- -- Topics reordering successful. All variables saved to file')
            return 1
        except:
            self._logger.info(
                '-- -- Topics reordering generated an error. Operation failed')
            return 0

    def resetTM(self):
        self._alphas_orig = np.load(self._TMfolder.joinpath('alphas_orig.npy'))
        self._betas_orig = np.load(self._TMfolder.joinpath('betas_orig.npy'))
        self._thetas_orig = sparse.load_npz(
            self._TMfolder.joinpath('thetas_orig.npz'))
        self._load_vocab()

        try:
            self.create(betas=self._betas_orig, thetas=self._thetas_orig,
                        alphas=self._alphas_orig, vocab=self._vocab)
            return 1
        except:
            return 0

    def recalculate_cohrs(self):

        self.load_tpc_descriptions()

        try:
            self.calculate_topic_coherence()

            self._save_cohr()

            self._logger.info(
                '-- -- Topics cohrence recalculation successful. All variables saved to file')
            return 1
        except:
            self._logger.info(
                '-- -- Topics cohrence recalculation  an error. Operation failed')
            return 0

    def to_dataframe(self):
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_betas_ds()
        self._load_topic_entropy()
        self._load_topic_coherence()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_ndocs_active()
        self._load_vocab()
        self._load_vocab_dicts()

        data = {
            "betas": [self._betas],
            "alphas": [self._alphas],
            "topic_entropy": [self._topic_entropy],
            "topic_coherence": [self._topic_coherence],
            "ndocs_active": [self._ndocs_active],
            "tpc_descriptions": [self._tpc_descriptions],
            "tpc_labels": [self._tpc_labels],
        }
        df = pd.DataFrame(data)
        return df, self._vocab_id2w


##############################################################################
#                                  MAIN                                      #
##############################################################################
if __name__ == "__main__":

    from tm_utils import look_for_path

    parser = argparse.ArgumentParser(
        description="Scripts for Topic Modeling Service")
    parser.add_argument("--path_TMmodels", type=str,
                        default=None, required=True,
                        metavar=("path_to_TMs"),
                        help="path to topic models folder")
    parser.add_argument("--listTMmodels", action="store_true", default=False,
                        help="List Available Topic Models")
    parser.add_argument("--getTMmodel", type=str, default=None,
                        metavar=("modelName"),
                        help="Get more info about a specific model")
    parser.add_argument("--deleteTMmodel", type=str, default=None,
                        metavar=("modelName"),
                        help="Delete Topic Model with selected name")
    parser.add_argument("--renameTM", type=str, default=None, nargs=2,
                        metavar=("modelName", "new_modelName"),
                        help="Rename Topic Model with selected name to new name")
    parser.add_argument("--copyTM", type=str, default=None, nargs=2,
                        metavar=("modelName", "new_modelName"),
                        help="Make a copy of Topic Model")
    parser.add_argument("--showTopics", type=str, default=None,
                        metavar=("modelName"),
                        help="Retrieve topic labels and word composition for selected model")
    parser.add_argument("--showTopicsAdvanced", type=str, default=None,
                        metavar=("modelName"),
                        help="Retrieve topic labels, word composition for selected model and advanced statistics")
    parser.add_argument("--setTpcLabels", type=str, default=None,
                        metavar=("modelName"),
                        help="Set Topics Labels for selected model")
    parser.add_argument("--deleteTopics", type=str, default=None,
                        metavar=("modelName"),
                        help="Remove topics from selected model")
    parser.add_argument("--getSimilarTopics", type=str, default=None,
                        metavar=("modelName"),
                        help="Retrieve information about similar topics for selected model")
    parser.add_argument("--fuseTopics", type=str, default=None,
                        metavar=("modelName"),
                        help="Merge topics from selected model")
    parser.add_argument("--sortTopics", type=str, default=None,
                        metavar=("modelName"),
                        help="Sort topics according to size")
    parser.add_argument("--resetTM", type=str, default=None,
                        metavar=("modelName"),
                        help="Reset Topic Model to its initial values after training")

    args = parser.parse_args()

    tmm = TMManager()

    tm_path = Path(args.path_TMmodels)

    if args.listTMmodels:
        allTMmodels = tmm.listTMmodels(tm_path)
        sys.stdout.write(json.dumps(allTMmodels))

    if args.getTMmodel:
        tm_path = look_for_path(tm_path, f"{args.getTMmodel}")
        allTMmodels = tmm.getTMmodel(tm_path.joinpath(f"{args.getTMmodel}"))
        sys.stdout.write(json.dumps(allTMmodels))

    if args.deleteTMmodel:
        tm_path = look_for_path(tm_path, f"{args.deleteTMmodel}")
        status = tmm.deleteTMmodel(tm_path.joinpath(f"{args.deleteTMmodel}"))
        sys.stdout.write(str(status))

    if args.renameTM:
        tm_path = look_for_path(tm_path, f"{args.renameTM[0]}")
        status = tmm.renameTMmodel(
            tm_path.joinpath(f"{args.renameTM[0]}"),
            tm_path.joinpath(f"{args.renameTM[1]}"),
        )
        sys.stdout.write(str(status))

    if args.copyTM:
        tm_path = look_for_path(tm_path, f"{args.copyTM[0]}")
        status = tmm.copyTMmodel(
            tm_path.joinpath(f"{args.copyTM[0]}"),
            tm_path.joinpath(f"{args.copyTM[1]}"),
        )
        sys.stdout.write(str(status))

    if args.showTopics:
        tm_path = look_for_path(tm_path, f"{args.showTopics}")
        tm = TMmodel(tm_path.joinpath(
            f"{args.showTopics}").joinpath('TMmodel'))
        sys.stdout.write(json.dumps(tm.showTopics()))

    if args.showTopicsAdvanced:
        tm_path = look_for_path(tm_path, f"{args.showTopicsAdvanced}")
        tm = TMmodel(tm_path.joinpath(
            f"{args.showTopicsAdvanced}").joinpath('TMmodel'))
        sys.stdout.write(json.dumps(tm.showTopicsAdvanced()))

    if args.setTpcLabels:
        # Labels should come from standard input
        TpcLabels = "".join([line for line in sys.stdin])
        TpcLabels = json.loads(TpcLabels.replace('\\"', '"'))
        tm_path = look_for_path(tm_path, f"{args.setTpcLabels}")
        tm = TMmodel(tm_path.joinpath(
            f"{args.setTpcLabels}").joinpath('TMmodel'))
        status = tm.setTpcLabels(TpcLabels)
        sys.stdout.write(str(status))

    if args.deleteTopics:
        # List of topics to remove should come from standard input
        tpcs = "".join([line for line in sys.stdin])
        tpcs = json.loads(tpcs.replace('\\"', '"'))
        tm_path = look_for_path(tm_path, f"{args.deleteTopics}")
        tm = TMmodel(tm_path.joinpath(
            f"{args.deleteTopics}").joinpath('TMmodel'))
        status = tm.deleteTopics(tpcs)
        sys.stdout.write(str(status))

    if args.getSimilarTopics:
        # List of topics to remove should come from standard input
        npairs = "".join([line for line in sys.stdin])
        npairs = json.loads(npairs.replace('\\"', '"'))
        tm_path = look_for_path(tm_path, f"{args.getSimilarTopics}")
        tm = TMmodel(tm_path.joinpath(
            f"{args.getSimilarTopics}").joinpath('TMmodel'))
        sys.stdout.write(json.dumps(tm.getSimilarTopics(int(npairs))))

    if args.fuseTopics:
        # List of topics to merge should come from standard input
        tpcs = "".join([line for line in sys.stdin])
        tpcs = json.loads(tpcs.replace('\\"', '"'))
        tm_path = look_for_path(tm_path, f"{args.fuseTopics}")
        tm = TMmodel(tm_path.joinpath(
            f"{args.fuseTopics}").joinpath('TMmodel'))
        status = tm.fuseTopics(tpcs)
        sys.stdout.write(str(status))

    if args.sortTopics:
        tm_path = look_for_path(tm_path, f"{args.sortTopics}")
        tm = TMmodel(tm_path.joinpath(
            f"{args.sortTopics}").joinpath('TMmodel'))
        status = tm.sortTopics()
        sys.stdout.write(str(status))

    if args.resetTM:
        tm_path = look_for_path(tm_path, f"{args.resetTM}")
        tm = TMmodel(tm_path.joinpath(f"{args.resetTM}").joinpath('TMmodel'))
        status = tm.resetTM()
        sys.stdout.write(str(status))
