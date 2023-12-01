"""
Classification Utils built over Cohere API

Raises:
    ValueError: In cases of invalid test counts or other size mismatches
"""
from pathlib import Path
from typing import List, Tuple

import cohere
import pandas as pd
from cohere.classify import Example
from tqdm.notebook import tqdm

Path.ls = lambda x: list(x.iterdir())  # type: ignore


class CohereResponse:
    """
    Utils for working with the Cohere Response
    """

    @staticmethod
    def parse_response(
        response: cohere.classify.Classifications,
    ) -> Tuple[List[str], List[float]]:
        """
        Parse a Cohere Response into a tuple of (labels, scores)

        Args:
            response ():

        Returns:
            Tuple[List[str], List[float]]: _description_
        """
        labels = []
        scores = []
        for classification in response.classifications:
            lbl = classification.prediction
            score = classification.labels[classification.prediction].confidence
            labels.append(lbl)
            scores.append(score)
        return labels, scores


class FewShotClassify(CohereResponse):
    """
    Data Management for Few Shot Classification using the Cohere API

    Inherits:
        CohereBase (CohereBase):  Base Utils for the Cohere API
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cohere_client: cohere.Client,
        train_counts: List[int] = [4, 8, 16, 32],
        test_count: int = 64,
        x_label: str = "Name",
        y_label: str = "Key",
    ) -> None:
        # pylint: disable=too-many-arguments, dangerous-default-value
        """
        Initialize the Few Shot Classification Data Manager

        Args:
            df (pd.DataFrame): Dataframe with all tagged samples. This will be split into train and test as needed
            co (cohere.Client): cohere Client object
            train_counts (List[int], optional): Defaults to [4, 8, 16, 32].
            test_count (int, optional): Number of test samples. Defaults to 64.
            x_label (str, optional): text column. Defaults to "Name".
            y_label (str, optional): label column. Defaults to "Key".

        Returns:
            None
        """
        self.df, self.train_counts, self.test_count, self.x_label, self.y_label = (
            df,
            train_counts,
            test_count,
            x_label,
            y_label,
        )
        self.cohere_client = cohere_client
        self.labels = list(self.df[y_label].unique())
        self.random_state = 37
        self.train_dfs, self.test_df = self.train_test_split(
            self.train_counts, self.test_count, self.labels
        )

    def __repr__(self) -> str:
        return f"FewShotClassify(Train Counts: {self.train_counts}, Test Count: {self.test_count}, Text Column(x_label): {self.x_label}, Target Column(y_label): {self.y_label})"

    @staticmethod
    def _compare(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        # Get all diferent values
        df3 = pd.merge(df1, df2, how="outer", indicator="Exist")
        df3 = df3.loc[df3["Exist"] != "both"]
        return df3

    def train_test_split(
        self, train_counts: List[int], test_count: int, labels: List[str]
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """
        Make train dataframes for each label, keep a test_df as well

        Args:
            train_counts (List[int]): Number of examples to keep for training for each label
            test_count (int): Number of examples to keep for test for each label
            labels (List[str]): List of labels to keep

        Raises:
            ValueError:

        Returns:
            Tuple[List[pd.DataFrame], pd.DataFrame]: List of train dataframes, test dataframe
        """
        train_dfs = []
        for n in train_counts:
            trn = []
            test_lbl_cuts = []
            for lbl in labels:
                class_cut = self.df[self.df[self.y_label] == lbl]
                if len(class_cut) < n + test_count:
                    raise ValueError(
                        f"Label {lbl} has only {len(class_cut)} samples, need {n + test_count}"
                    )
                # print(f"Label: {lbl}, Class Cut has {len(class_cut)} samples")
                test_cut = class_cut.sample(test_count, random_state=self.random_state)
                # print(f"Label: {lbl}, Test Cut has {len(test_cut)} samples")
                test_lbl_cuts.append(test_cut)
                left_over = FewShotClassify._compare(test_cut, class_cut)
                # print(f"Label: {lbl}, Left Over has {len(left_over)} samples")
                if len(left_over) < n:
                    raise ValueError(
                        f"For label '{lbl}' insufficient number of training samples left: {len(left_over)} <= {n}"
                    )
                trn.append(left_over.sample(n, random_state=self.random_state))
            train_dfs.append(pd.concat(trn))
        test_df = pd.concat(test_lbl_cuts)
        return train_dfs, test_df

    def make_examples(self, df: pd.DataFrame) -> List[Example]:
        """
        #TODO: Iterative and slow, make this parallel and fast
        """
        examples = []
        for row in df.iterrows():
            text = row[1][self.x_label]
            lbl = row[1][self.y_label]
            examples.append(Example(text, lbl))
        return examples

    def predict(self) -> List[cohere.classify.Classifications]:
        """
        Predict on the test set

        Returns:
            cohere.response: COHERE_RESPONSE
        """
        responses = []
        for trn_df in tqdm(self.train_dfs):
            inputs = self.test_df[self.x_label].tolist()
            examples = self.make_examples(trn_df)
            response = self.cohere_client.classify(inputs=inputs, examples=examples)
        responses.append(response)
        return responses
