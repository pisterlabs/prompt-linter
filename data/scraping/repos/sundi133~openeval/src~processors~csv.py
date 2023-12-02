import pandas as pd
import os
import io
import logging
import json
from typing import List
from langchain.chains import LLMChain
from .basefile import DataProcessor

# TODO move to central logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


class CSVProcessor(DataProcessor):
    def __init__(self, data_path: str) -> None:
        super().__init__(data_path)
        self.file_extension = os.path.splitext(data_path)[-1].lower()
        self.data = self.parse()
        self.qa_dict = {}
        self.qa_array = []
        self.schema = None

    def parse(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, index_col=False)
        self.schema = list(df.columns)
        df.fillna("", inplace=True)
        return df

    def get_randomized_samples(
        self,
        df: pd.DataFrame,
        sample_size: int,
        products_group_size: int,
        group_columns: List[str],
    ) -> pd.DataFrame:
        if len(group_columns) == 0:
            if sample_size > df.shape[0]:
                sample_size = df.shape[0]
            return df.sample(n=sample_size, random_state=42)
        # Group the group_columns
        grouped = df.groupby(group_columns)

        # Define a filter function to check if the group has at least 'products_group_size' products
        def filter_groups(group):
            return len(group) >= products_group_size

        # Apply the filter function to each group and concatenate the results
        filtered_df = grouped.filter(filter_groups)

        # Calculate group counts after filtering
        group_counts = (
            filtered_df.groupby(group_columns).size().reset_index(name="count")
        )

        # Filter groups with at least 'products_group_size' products
        group_counts_filter = group_counts[group_counts["count"] >= products_group_size]

        # Randomly select 'sample_size' groups
        if sample_size > group_counts_filter.shape[0]:
            sample_size = group_counts_filter.shape[0]
        randomized_grouping = group_counts_filter.sample(n=sample_size, random_state=42)
        return randomized_grouping

    def generate_qa_pairs(
        self,
        randomized_samples: pd.DataFrame,
        df: pd.DataFrame,
        sample_size: int,
        products_group_size: int,
        group_columns: List[str],
        number_of_questions: int,
        qa_generator: LLMChain,
    ) -> None:
        logger.info(
            {
                "message": "Generating question & answer pairs",
                "randomized_samples": randomized_samples.shape,
            }
        )
        for _index, group_row in randomized_samples.iterrows():
            filtered_dataframes = []
            group_filters = []
            logger.info(
                {
                    "message": "Generating question",
                    "_index": _index,
                    "group_columns": group_columns,
                    "group_row": group_row.shape,
                    "df": df.shape,
                    "df_columns": df.columns,
                }
            )
            logger.info("****")
            logger.info(group_row.head())
            # Create a filter for the current group
            for column in group_columns:
                # Create a filter condition for the current column and group_row
                condition = df[column] == group_row[column]

                # Append the condition to the group_filters list
                group_filters.append(condition)

            # Combine all the filter conditions using the "&" operator
            if group_filters:
                group_filter = pd.DataFrame(group_filters).all(axis=0)
            else:
                # Handle the case where there are no conditions in group_filters
                group_filter = pd.Series(True, index=df.index)

            filtered_df = df[group_filter]

            filtered_dataframes.append(filtered_df)

            # Combine the filtered DataFrames into a single DataFrame
            combined_filtered_df = pd.concat(filtered_dataframes, ignore_index=True)

            # Initialize a CSV buffer for writing
            csv_buffer = io.StringIO()

            # Write the DataFrame to the CSV buffer
            combined_filtered_df.to_csv(csv_buffer, index=False, header=True)

            # Get the CSV string from the buffer
            records = csv_buffer.getvalue()

            # Close the buffer (optional)
            csv_buffer.close()

            qa_pair = qa_generator.run(
                products=records,
                number_of_questions=number_of_questions,
                schema=self.schema,
            )

            # Log generated questions
            logger.info(
                {
                    "message": "Generated question & answer pair",
                    "questions": qa_pair,
                }
            )

            # Split questions by newline and process each question
            question_array = json.loads(qa_pair)

            for record in question_array:
                # Log each generated question
                logger.info(
                    {
                        "message": "Generated question",
                        "question_answer": record,
                    }
                )
                self.add_output_sample(record)
        return self.qa_dict

    def add_output_sample(self, record: json) -> None:
        self.qa_array.append(
            {
                "question": record["question"],
                "answer": record["answer"],
            }
        )

    def write(self, file_path: str) -> None:
        with open(file_path, "w") as output_file:
            json.dump(self.qa_array, output_file, indent=4)
