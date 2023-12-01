import pandas as pd
import uuid
import os
import re
import io
import numpy as np
import pandas as pd
from ._prompt_builder import PromptBuilder
from ._test_tree_browser import TestTreeBrowser
from ._topic_model import TopicLabelingModel, TopicMembershipModel
from .embedders import unique
from .generators import OpenAI
import adatest

class TestTree():
    """ A hierarchically organized set of tests represented as a DataFrame.

    This represents a hierarchically organized set of tests that all target a specific class of models (such as sentiment
    analysis models, or translation models). To interact with a test tree you can use either the `__call__` method to
    view and create tests directly in a Jupyter notebook, or you can call the `serve` method to launch a standalone
    webserver. A TestTree object also conforms to most of the standard pandas DataFrame API.
    """

    def __init__(self, tests=None, index=None, labeling_model=TopicLabelingModel, membership_model=TopicMembershipModel, compute_embeddings=False, ensure_topic_markers=True, duplicate_threshold=0.92, disable_topic_model=False, **kwargs):
        """ Create a new test tree.

        Parameters
        ----------
        tests : str or DataFrame or list or tuple or None
            The tests to load as a test tree. If a string is provided, it is assumed to be a path to a CSV file containing
            the tests. If tests is a tuple of two elements, it is assumed to be a dataset of (data, labels) which will be used to build a test tree.
            Otherwise tests is passed to the pandas DataFrame constructor to load the tests as a DataFrame.

        index : list or list-like or None
            Assigns an index to underlying tests frame, or auto generates if not provided.

        compute_embeddings: boolean
            If True, use the global adatest.embed to build embeddings of tests in the TestTree.

        duplicate_threshold: float in [0, 1]
            Used for deduplication. If two tests have cosine similarity >= duplicate_threshold, they will be marked as duplicates.
        """

        # the canonical ordered list of test tree columns
        column_names = ['topic', 'input', 'output', 'label', 'labeler', 'description']

        self.labeling_model = labeling_model
        self.membership_model = membership_model
        self.disable_topic_model = disable_topic_model
        self.duplicate_threshold = duplicate_threshold

        # create a new test tree in memory
        if tests is None:
            self._tests = pd.DataFrame([], columns=column_names, dtype=str)
            self._tests_location = None

        # create a new test tree on disk (lazily saved)
        elif isinstance(tests, str) and not os.path.isfile(tests):
            self._tests = pd.DataFrame([], columns=column_names)
            self._tests_location = tests

        # load the test tree from a file or IO stream
        elif isinstance(tests, str) or isinstance(tests, io.TextIOBase):
            self._tests_location = tests
            if os.path.isfile(tests) or isinstance(tests, io.TextIOBase):
                self._tests = pd.read_csv(tests, index_col=0, dtype=str, keep_default_na=False)
                self._tests.index = self._tests.index.map(str)
            else:
                raise Exception(f"The provided tests file is not supported: {tests}")

        # tests is a pandas-like object, e.g. a subset called by the TestTreeLocIndexer
        else:
            if index is None:
                index = [uuid.uuid4().hex for _ in range(len(tests))]
            self._tests = pd.DataFrame(tests, **kwargs)
            self._tests.index = index
            self._tests_location = None

        # ensure we have required columns
        for c in ["input", "output", "label"]:
            if c not in self._tests.columns:
                raise Exception("The test tree being loaded must contain a '"+c+"' column!")

        # fill in any other missing columns
        if "input_display" not in self._tests.columns:
            self._tests["input_display"] = self._tests["input"]
        for column in ["topic", "description", "create_time"]:
            if column not in self._tests.columns:
                self._tests[column] = ["" for _ in range(self._tests.shape[0])]
        if "labeler" not in self._tests.columns:
            self._tests["labeler"] = ["imputed" for _ in range(self._tests.shape[0])]
        if "label_confidence" not in self._tests.columns:
            self._tests["label_confidence"] = [1.0 for _ in range(self._tests.shape[0])]

        # ensure that all topics have a topic_marker entry
        if ensure_topic_markers:
            self.ensure_topic_markers()
            
        # sanitize all topic names
        self._tests['topic'] = self._tests['topic'].apply(adatest.utils.sanitize_topic_name)

        # drop any duplicate index values
        self._tests = self._tests.groupby(level=0).first()
        
        # drop any duplicate rows
        self._tests.drop_duplicates(["topic", "input", "output", "labeler"], inplace=True)

        # put the columns in a consistent order
        self._tests = self._tests[column_names + [c for c in self._tests.columns if c not in column_names]]

        if compute_embeddings:
            self._cache_embeddings()

        self._topic_labeling_models = {}
        self._topic_membership_models = {}

    @property
    def name(self):
        return re.split(r"\/", self._tests_location)[-1] if self._tests_location is not None else "Tests"

    def ensure_topic_markers(self):
        marked_topics = {t: True for t in set(self._tests.loc[self._tests["label"] == "topic_marker"]["topic"])}
        for topic in set(self._tests["topic"]):
            parts = topic.split("/")
            for i in range(1, len(parts)+1):
                parent_topic = "/".join(parts[:i])
                if parent_topic not in marked_topics:
                    self._tests.loc[uuid.uuid4().hex] = {
                        "label": "topic_marker",
                        "topic": parent_topic,
                        "labeler": "imputed",
                        "input": "",
                        "input_display": "",
                        "output": "",
                        "description": ""
                    }
                    marked_topics[parent_topic] = True

    def __getitem__(self, key):
        """ TestSets act just like a DataFrame when sliced. """
        subset = self._tests[key]
        if hasattr(subset, 'columns') and len(set(["topic", "input", "output", "label"]) - set(subset.columns)) == 0:
            return self.__class__(subset, index=subset.index)
        return subset

    def __setitem__(self, key, value):
        """ TestSets act just like a DataFrame when sliced, including assignment. """
        self._tests[key] = value

    # all these methods directly expose the underlying DataFrame API
    @property
    def loc(self):
        return TestTreeLocIndexer(self)
    @property
    def iloc(self):
        return TestTreeILocIndexer(self)
    @property
    def index(self):
        return self._tests.index
    @property
    def columns(self):
        return self._tests.columns
    @property
    def shape(self):
        return self._tests.shape
    @property
    def str(self):
        return self._tests.str
    @property
    def iterrows(self):
        return self._tests.iterrows
    @property
    def groupby(self):
        return self._tests.groupby
    @property
    def drop(self):
        return self._tests.drop
    @property
    def insert(self):
        return self._tests.insert
    @property
    def copy(self):
        return self._tests.copy
    @property
    def sort_values(self):
        return self._tests.sort_values
    
    # NOTE: Can't delegate to df.append as it is deprecated in favor of pd.concat, which we can't use due to type checks 
    def append(self, test_tree, axis=0):
        if isinstance(test_tree, pd.DataFrame):
            self._tests = pd.concat([self._tests, test_tree], axis=axis)
        elif isinstance(test_tree, TestTree):
            self._tests = pd.concat([self._tests, test_tree._tests], axis=axis)
        elif isinstance(test_tree, dict):
            # check if the values are strings or lists of strings
            if any([isinstance(v, str) for v in test_tree.values()]):
                self._tests = pd.concat([self._tests, pd.DataFrame({k: [test_tree[k]] for k in test_tree}, index=[uuid.uuid4().hex])], axis=axis)
            else:
                self._tests = pd.concat([self._tests, pd.DataFrame(test_tree)], axis=axis)
        return None

    def __len__(self):
        return self._tests.__len__()
    
    def __setitem__(self, key, value):
        return self._tests.__setitem__(key, value)
    
    def to_csv(self, file=None):
        no_suggestions = self._tests.loc[["/__suggestions__" not in topic for topic in self._tests["topic"]]]
        if file is None:
            no_suggestions.to_csv(self._tests_location)
        else:
            no_suggestions.to_csv(file)

    def get_children_in_topic(self, topic, include_self=False, include_tests=True, include_topics=True, direct_children_only=False, include_suggestions=False):
        """ Return a subset of the test tree containing only tests that match the given topic.

        Parameters
        ----------
        topic : str
            The topic to filter the test tree by.

        include_self : bool
            Whether to include the row corresponding to the topic itself
        
        include_topics : bool
            Whether to include rows marked as "topic_marker"

        direct_children_only : bool
            Whether to only return direct children of the topic, i.e., all direct tests and direct subtopics.

        include_suggestions : bool
            Whether to include suggestions
        """
        children = self._tests.apply(
            lambda row: adatest.utils.is_subtopic(topic, row["topic"]) \
                and (include_topics or row["label"] != "topic_marker") \
                and (include_tests or row["label"] == "topic_marker") \
                and ((not direct_children_only) \
                        or (topic == row["topic"]) \
                        or (row["topic"][len(topic)+1:].count("/") == 0 and row["label"] == "topic_marker") \
                        or (row["topic"][len(topic)+1:].startswith("__suggestions__")) \
                    ) \
                and (include_suggestions or '__suggestions__' not in row["topic"]),
            axis=1
        )
        if not include_self:
            self_ix = self._tests.index[(self._tests["topic"] == topic) & (self._tests["label"] == "topic_marker")]
            children.loc[self_ix] = False
        
        if len(children) == 0: return children 
        else: return self.loc[children] 

    def get_topics(self):
        """Return a Series of topic uuids and their names"""
        return self.get_children_in_topic("", include_tests=False, include_self=True)['topic']

    def get_topic_id(self, topic):
        """
        Returns the id of the topic.
        Returns None if not found.
        """
        indices = self._tests.index[(self._tests['topic'] == topic) & (self._tests['label'] == 'topic_marker')]
        # if for some reason, there are two indices for this topic, return the first one
        return indices.tolist()[0] if len(indices) > 0 else None

    def topic_has_direct_tests(self, target_topic: str)-> bool:
        """Check if a topic has direct tests."""
        hdt_df = self._tests.apply(
            lambda row: row['topic']==target_topic and row['label'] != 'topic_marker',
            axis=1
        )
        return len(hdt_df) > 0 and hdt_df.any()

    def topic_has_subtopics(self, target_topic: str) -> bool:
        """Check if a topic has subtopics."""
        has_subtopics_df = self._tests.apply(
            lambda row: row['topic']!=target_topic and adatest.utils.is_subtopic(target_topic, row["topic"]),
            axis=1
        )
        return len(has_subtopics_df) > 0 and has_subtopics_df.any()

    def filter_tree(self, filter_fn):
        """Return a copy of the tree with the filter_fn applied"""
        matches = self._tests.apply(
            filter_fn,
            axis=1
        )
        if len(matches) == 0: return matches 
        else: return self.loc[matches] 

    def drop_topic(self, topic):
        """ Remove a topic from the test tree. """
        self._tests = self._tests.loc[self._tests["topic"] != topic]

    def adapt(self, scorer=None, generator=OpenAI(), auto_save=False, user="anonymous",
              max_suggestions=20, suggestion_thread_budget=0, prompt_builder=PromptBuilder(), active_generator="default",
              disable_topic_suggestions=False):
        """ 
        Set up a test tree browser to enable testing a model (wrapped in a Scorer) for this test tree.
        The TestTreeBrowser object that can be used to
        browse the tree and add new tests to adapt it to the target model.
        
        Parameters
        ----------
        scorer : adatest.Scorer or callable
            The scorer (that wraps a model) to used to score the tests. If a function is provided, it will be wrapped in a scorer.
            Passing a dictionary of scorers will score multiple models at the same time. Note that the models are expected to take
            a list of strings as input, and output either a classification probability vector or a string.

        generator : adatest.Generator or dict[adatest.Generators]
            A source to generate new tests from. Currently supported generator types are language models, existing test trees, or datasets.

        auto_save : bool
            Whether to automatically save the test tree after each edit.

        user : str
            The user name to author new tests with.

        max_suggestions : int
            The maximum number of suggestions to generate each time the user asks for test suggestions.

        suggestion_thread_budget : float
            This controls how many parallel suggestion processes to use when generating suggestions. A value of 0 means we create
            no parallel threads (i.e. we use a single thread), 0.5 means we create as many parallel threads as possible without
            increase the number of tokens we process by more than 50% (1.5 would mean 150%, etc.). Each thread process will use a
            different randomized LM prompt for test generation, so more threads will result in more diversity, but come at the cost
            of reading more prompt variations.

        prompt_builder : adatest.PromptBuilder
            A prompt builder to use when generating prompts for new tests. This object controls how the LM prompts
            are created when generating new tests.

        active_generator : "default", or a key name if generators is a dictionary
            Which generator from adatest.generators to use when generating new tests. This should always be set to "default" if
            generators is just a single generator and not a dictionary of generators.
        """

        # build the test tree browser
        return TestTreeBrowser(
            self,
            scorer=scorer,
            generators=generator,
            auto_save=auto_save,
            user=user,
            max_suggestions=max_suggestions,
            suggestion_thread_budget=suggestion_thread_budget,
            prompt_builder=prompt_builder,
            active_generator=active_generator,
            disable_topic_suggestions=disable_topic_suggestions,
        )

    def __repr__(self):
        return self._tests.__repr__()

    def _repr_html_(self):
        return self._tests._repr_html_()

    def deduplicate_subtopics(self, topic):
        """ Remove duplicate subtopic suggestions within a topic from the test tree.
        Only deduplicates by exact match.
        If two suggestions duplicate each other, we give preference to the first item.
        Returns the number of subtopic suggestions removed.
        """
        already_seen, drop_ids = set(), []
        subtopics = self.get_children_in_topic(topic, include_suggestions=True, include_self=False, include_topics=True, direct_children_only=True)
        subtopics = subtopics[subtopics['label'].apply(lambda l: l == 'topic_marker').array]
        if len(subtopics) == 0: return 0
        suggestion_mask = subtopics['topic'].apply(lambda t: '/__suggestions__' in t).array

        # first, deduplicate by string input
        # go through non-suggestions first
        for id, test in subtopics.loc[~suggestion_mask].iterrows():
            already_seen.add(test.topic[len(topic)+1:].strip())

        # go through suggestions next
        for id, test in subtopics.loc[suggestion_mask].iterrows():
            name = test.topic[len(topic)+1:].replace('__suggestions__/', '').strip()
            if name in already_seen: drop_ids.append(id)
            else: already_seen.add(name)

        # drop the duplicates
        num_dropped = len(drop_ids)
        self._tests.drop(drop_ids, axis=0, inplace=True)        
        print(f"Removed {num_dropped} suggestions in deduplicate_subtopics()")
        return num_dropped

    def deduplicate_tests(self, topic):
        """ Remove duplicate test suggestions within a topic from the test tree.
        Deduplicates by both exact match of URL and image embedding similarity.
        If two suggestions duplicate each other, we give preference to the first item.
        Deduplicates against ALL pass/failed tests inside the tree.
        Returns the number of test suggestions removed.
        """
        already_seen, drop_ids = set(), []
        tests_in_topic = self.get_children_in_topic(topic, include_suggestions=True, include_self=False, include_topics=False, direct_children_only=False)
        tests_outside_topic = self.get_children_in_topic(
            "", include_suggestions=False, include_self=False, include_topics=False, direct_children_only=False
        ).filter_tree(lambda row: row['label'] in ('pass', 'fail')) # leave off_topic images from other folders
        suggestion_mask = tests_in_topic['topic'].apply(lambda t: '/__suggestions__' in t).array

        # no suggestions, and no tests
        if len(tests_in_topic) == 0: return 0

        # first, deduplicate by string input
        # go through non-suggestions first
        for id, test in tests_in_topic.loc[~suggestion_mask].iterrows():
            already_seen.add(test.input)
        for id, test in tests_outside_topic.iterrows():
            already_seen.add(test.input)

        # go through suggestions next
        for id, test in tests_in_topic.loc[suggestion_mask].iterrows():
            if test.input in already_seen: drop_ids.append(id)
            else: already_seen.add(test.input)

        # drop the duplicates
        num_dropped = len(drop_ids)
        self._tests.drop(drop_ids, axis=0, inplace=True)
        tests_in_topic.drop(drop_ids, axis=0, inplace=True)
        suggestion_mask = tests_in_topic['topic'].apply(lambda t: '/__suggestions__' in t).array

        # more aggressive deduplication by comparing image embeddings
        non_sugg_embs = adatest.embed(
            tests_in_topic.loc[~suggestion_mask]['input'].tolist()
            + tests_outside_topic['input'].tolist()
        )
        sugg_embs = adatest.embed(tests_in_topic.loc[suggestion_mask]['input'].tolist())
        _, ix = unique(non_sugg_embs + sugg_embs, return_index=True, threshold=self.duplicate_threshold)
        ix = ix[ix > len(non_sugg_embs)] - len(non_sugg_embs)
        keep_ids = tests_in_topic.loc[suggestion_mask].index[ix]
        drop_ids = list(set(tests_in_topic.loc[suggestion_mask].index) - set(keep_ids))
        num_dropped += len(drop_ids)
        self._tests.drop(drop_ids, axis=0, inplace=True)
        
        print(f"Removed {num_dropped} suggestions in deduplicate_tests()")
        return num_dropped

    def validate_input_displays(self, static_dir):
        """ 
        Check all test.input_display fields for validity; replace invalid test.input_displays with test.input
        """
        def _validate_input(test):
            input_display, input = test['input_display'], test['input']
            if type(input_display) != str or input_display == "": return input
            if input_display.startswith("__IMAGE=/_static") \
                and static_dir is not None \
                and not os.path.exists(input_display.replace("__IMAGE=/_static", static_dir)):
                    return input
            return input_display
        if len(self._tests) == 0: return
        self._tests['input_display'] = self._tests.apply(_validate_input, axis=1)

    def _cache_embeddings(self, ids=None):
        """Cache the embeddings of ids OR all uncached (new) tests/topics."""
        if ids is None:
            ids = self._tests.index

        # see what new embeddings we need to compute
        to_embed = []
        for id in ids:
            test = self._tests.loc[id]
            if test.label == "topic_marker":
                to_embed.append(adatest.utils.pretty_topic_name(test.topic))
            else:
                # since the topic model uses both the input and output strings, we embed both
                to_embed.extend([test.input, test.output])
            
        adatest.embed(to_embed)

    def get_topic_model_labels(self):
        """Impute missing labels in the test tree:
        1. Run the topic models for all new suggestions.
        2. Mark all topic markers as topic_marker
        """
        ids_to_impute = self._tests.index[self._tests["label"] == ""]
        self._cache_embeddings(ids_to_impute)
        for id in ids_to_impute:
            test = self._tests.loc[id]
            self._tests.loc[id, "labeler"] = "imputed"

            if not self.disable_topic_model:
                # compute membership
                membership_pred, membership_conf = self.topic_membership_model(test.topic)(test.input, return_confidence=True)
                if membership_pred == "off_topic":
                    self._tests.loc[id, "label"] = "off_topic"
                    self._tests.loc[id, "label_confidence"] = membership_conf
                    continue
                # compute label
                label_pred, label_conf = self.topic_labeling_model(test.topic)(test.input, test.output, return_confidence=True)
                self._tests.loc[id, "label"] = label_pred
                self._tests.loc[id, "label_confidence"] = label_conf
            
            else:
                self._tests.loc[id, "label"] = "Unknown"
                self._tests.loc[id, "label_confidence"] = 0

    def topic_labeling_model(self, topic):
        topic = topic.replace("/__suggestions__", "") # predict suggestions using their parent topic label model
        if topic not in self._topic_labeling_models:
            self._topic_labeling_models[topic] = self.labeling_model(topic, self)
        return self._topic_labeling_models[topic]

    def topic_membership_model(self, topic):
        topic = topic.replace("/__suggestions__", "") # predict suggestions using their parent topic membership model
        if topic not in self._topic_membership_models:
            self._topic_membership_models[topic] = self.membership_model(topic, self)
        return self._topic_membership_models[topic]

    def retrain_topic_labeling_model(self, topic):
        self._topic_labeling_models[topic] = self.labeling_model(topic, self)

    def retrain_topic_membership_model(self, topic):
        self._topic_membership_models[topic] = self.membership_model(topic, self)
    
    def get_scores(self, score_column):
        """
        Returns an array containing scores \in [0, 1] calculated from the given score_column and the test's label.
        The array contains one value for each row in the tree.

        Assumes that topic markers are ensured.
        
        TESTS
        - The test score is np.nan if it is labeled off_topic.
        - The test score is (1 - normalized_to_01(score_column))/2 if it is labeled as pass, s.t. range is [0.01, 0.5]
        - The test score is (1 + normalized_to_01(score_column))/2 if it is labeled as fail, s.t. range is [0.5, 1]
        We want to avoid returning 0 because in _prompt_builder.py, 0 is used to mark a row as not sample-able. 

        TOPICS
        After tests are scored, topThics are always scored as the average of their direct test scores.
        """
        values = self._tests[score_column].copy()
        
        # first, compute test scores
        mask = (self._tests["label"] != "topic_marker")
        values.loc[mask] = pd.to_numeric(values.loc[mask], errors="raise")

        # normalize to [0,1]. may occur with detection.
        if values.loc[mask].min() < 0 or values.loc[mask].max() > 1: 
            values.loc[mask] = (
                np.exp(values.loc[mask].to_numpy(dtype='float')) / np.exp(values.loc[mask].to_numpy(dtype='float')).sum()
        )

        # adjust based on label
        for id in self._tests.index[mask]:
            label = self._tests.loc[id, "label"]
            if  label == "off_topic": values.loc[id] = np.nan
            elif label == "pass": values.loc[id] = max((1 - values.loc[id])/2, 0.01)
            elif label == "fail": values.loc[id] = (1 + values.loc[id])/2
        
        # next, compute topic scores
        # if a topic has no tests, topic_scores will contain NaN
        # in the edge case of the test tree containing no topics besides /, topic_scores will be an empty df, so we set the scores to nan
        topic_scores = self._tests.loc[~mask].apply(
            lambda row: values.loc[self.get_children_in_topic(row["topic"], include_topics=False, direct_children_only=True).index].mean(),
            axis=1
        ) # may contain nans if no children
        values.loc[~mask] = topic_scores if not topic_scores.empty else np.nan
        return values.to_numpy()

class TestTreeLocIndexer():
    def __init__(self, test_tree):
        self.test_tree = test_tree

    def __repr__(self):
        return "TestTreeLocIndexer is an intermediate object for operating on TestTrees. Slice this object further to yield useful results."

    def __getitem__(self, key):
        subset = self.test_tree._tests.loc[key]
        
        # If all columns haven't changed, it's still a valid test tree
        if hasattr(subset, 'columns') and len(set(["topic", "input", "output", "label"]) - set(subset.columns)) == 0:
            test_tree_slice = TestTree(subset, index=subset.index, ensure_topic_markers=False)
            test_tree_slice._tests_location = self.test_tree._tests_location
            return test_tree_slice

        # If columns have been dropped (e.g., key is a tuple selecting both an id and a column), return a Pandas object
        else:
            return subset
    
    def __setitem__(self, key, value):
        self.test_tree._tests.loc[key] = value
    
class TestTreeILocIndexer():
    def __init__(self, test_tree):
        self.test_tree = test_tree

    def __repr__(self):
        return "TestTreeILocIndexer is an intermediate object for operating on TestTrees. Slice this object further to yield useful results."

    def __getitem__(self, key):
        # If all columns haven't changed, it's still a valid test tree
        # If columns have been dropped, return a Pandas object
        
        subset = self.test_tree._tests.iloc[key]
        if hasattr(subset, 'columns') and len(set(["topic", "input", "output", "label"]) - set(subset.columns)) == 0:
            test_tree_slice = TestTree(subset, ensure_topic_markers=False)
            test_tree_slice._tests_location = self.test_tree._tests_location
            return test_tree_slice
        else:
            return subset
    
    def __setitem__(self, key, value):
        self.test_tree._tests.iloc[key] = value