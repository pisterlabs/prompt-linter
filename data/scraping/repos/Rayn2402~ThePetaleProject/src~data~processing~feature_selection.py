"""
Filename: feature_selection.py

Author: Nicolas Raymond

Description: Defines feature selector object, that removes unimportant features

Date of last modification : 2022/3/30
"""

from numpy import zeros
from os.path import join
from pandas import DataFrame, concat
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.data.processing.datasets import PetaleDataset
from typing import List, Optional, Tuple, Union


class FeatureSelector:
    """
    Object in charge of selecting the most important features of the dataset.
    Inspired from OpenAI feature selection code.

    See the following source:
    Deep Learning for Coders with fastai & PyTorch : AI Applications Without a PhD (p.486-489)
    """
    RECORDS_FILE = "feature_selection_records.csv"

    def __init__(self,
                 threshold: List[float],
                 cumulative_imp: List[bool],
                 nb_iter: int = 5,
                 seed: Optional[int] = None):
        """
        Sets protected attributes

        Args:
            threshold: list of threshold values used to for each features group
            cumulative_imp: list of bool for each features group. If True, features will be selected until their cumulative importance
                            reach the threshold. Otherwise, all features with an importance below
                            the threshold will be removed.
            nb_iter: number of times the feature importance must be calculated before taking the average
            seed: number used as a random state for the random forest doing feature selection
        """
        for t in threshold:
            if not 0 < t <= 1:
                raise ValueError('The threshold must be in range (0, 1]')

        self.__cumulative_imp = cumulative_imp
        self.__nb_iter = nb_iter
        self.__seed = seed
        self.__thresh = threshold

    def __call__(self,
                 dataset: PetaleDataset,
                 records_path: Optional[str] = None,
                 return_imp: bool = False) -> Tuple:
        """
        Extracts most important features using a random forest

        Args:
            dataset: custom dataset
            records_path: paths used to store figures and importance table
            return_imp: if True, feature importance are also returned

        Returns: list of cont_cols preserved, list of cat_cols preserved
        """
        # Extract feature importance
        fi_table = self.get_features_importance(dataset)

        # Select the subset of selected feature
        selected_features = fi_table.loc[fi_table['status'] == 'selected', 'features'].values

        # Save selected cont_cols and cat_cols
        if dataset.cont_cols is not None:
            cont_cols = [c for c in dataset.cont_cols if c in selected_features]
        else:
            cont_cols = None

        if dataset.cat_cols is not None:
            cat_cols = [c for c in dataset.cat_cols if c in selected_features]
        else:
            cat_cols = None

        # Save records in a csv
        if records_path is not None:
            fi_table.to_csv(join(records_path, FeatureSelector.RECORDS_FILE), index=False)

        if return_imp:

            # We modify fi_dict format
            fi_dict = {row['features']: row['imp'] for _, row in fi_table.iterrows()}
            return cont_cols, cat_cols, fi_dict

        return cont_cols, cat_cols

    def get_features_importance(self, dataset: PetaleDataset) -> DataFrame:
        """
        Trains a random forest (with default sklearn hyperparameters) to solve the classification
        or regression problems and uses it to extract feature importance.

        Args:
            dataset: custom dataset

        Returns: Dataframe with feature importance
        """
        # Extraction of current training mask
        mask = dataset.train_mask

        # Classifier constructor
        model_constructor = RandomForestClassifier if dataset.classification else RandomForestRegressor

        # List of dataframe to stack
        fi_tables = []

        # Validation of attributes
        nb_groups = len(dataset.feature_selection_idx_groups.keys())
        if len(self.__thresh) != nb_groups:
            raise ValueError('The number of given thresholds is not equal to the number of feature groups')
        if len(self.__cumulative_imp) != nb_groups:
            raise ValueError('The number of given cumulative imp. bool is not equal to the number of feature groups')

        for i, group in enumerate(dataset.feature_selection_idx_groups.values()):

            thresh, use_cumulative_imp = self.__thresh[i], self.__cumulative_imp[i]
            features, idx = group['features'], group['idx']

            # Mean importance initialization
            imp = zeros(len(features))

            # Data extraction
            x, y, _ = dataset[mask]
            x = x[:, idx]

            for i in range(self.__nb_iter):

                # Random forest training
                model = model_constructor(n_jobs=-1, oob_score=True, random_state=self.__seed + i).fit(x, y)

                imp += model.feature_importances_

            # Creation of feature importance table
            fi_table = DataFrame({'features': features,
                                  'imp': imp/self.__nb_iter}).sort_values('imp', ascending=False)

            # Addition of a column that indicates if the features are selected
            if use_cumulative_imp:
                cumulative_imp = 0
                status_list = []
                for index, row in fi_table.iterrows():
                    cumulative_imp += row['imp']
                    status_list.append('selected')
                    if cumulative_imp > thresh:
                        break
                status_list += ['rejected']*(fi_table.shape[0] - len(status_list))
                fi_table['status'] = status_list
            else:
                fi_table['status'] = ['rejected']*fi_table.shape[0]
                fi_table.loc[fi_table['imp'] >= thresh, 'status'] = 'selected'

            fi_tables.append(fi_table)

        # We stack the dataframes
        fi_table = concat(fi_tables, axis=0, ignore_index=True)

        # Rounding of importance values
        fi_table['imp'] = fi_table['imp'].apply(lambda x: round(x, 4))

        return fi_table
