import dotenv
import guardrails as gd
import openai
import pandas as pd
from guardrails_specs.specs import rail_str
from token_validator import TokenValidator
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from scipy.cluster.hierarchy import linkage, fcluster

from rich import print as rprint
import os
import logging
import re

dotenv.load_dotenv()
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL_NAME")
MAX_TOKENS = 2000


def preprocess(text):
    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Convert to lowercase
    text = text.lower().strip()

    return text


class TransactionParser:
    def __init__(self, transaction_string: str):
        self.transaction_string = transaction_string
        self.parsed_transactions = None
        self.llm = None
        self.token_validator = None
        self.transaction_categories = None
        self.last_raw_response_from_guard_completion = None
        logging.basicConfig(
            filename="llm_cals.log",
            level=logging.INFO,
            force=True,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(
            __name__
        )  # Gets or creates a logger with the name of the current module

    def start_llm(self):
        self.llm = OpenAI(temperature=0)

        self.logger.info("Creating an instance of LLM - Transaction Parser")

    def get_guard_chat_completion(self, input: str, guard: gd.Guard, token_number: int):
        self.logger.info(
            f"Making LLM call with {token_number} max_tokens, for input:  {input}"
        )

        raw_response, validated_response = guard(
            openai.ChatCompletion.create,
            prompt_params={"transaction_string": input},
            model=MODEL,
            max_tokens=token_number,
            temperature=0.0,
        )

        self.last_raw_response_from_guard_completion = raw_response

        self.logger.info(
            f"LLM Call completed with the following response paramters: \n {raw_response}"
        )

        return validated_response

    def get_chat_completion(self, prompt: str):
        self.logger.info(f"Making LLM call with for input:  {prompt}")

        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        self.logger.info(
            f"LLM Call completed with the following response paramters: \n {response['usage']}"
        )

        self.last_response_chat_completion = response

        return response["choices"][0]["message"]["content"]

    def parse_transactions(self):
        guard = gd.Guard.from_rail_string(rail_str)
        openai.api_key = OPEN_API_KEY

        self.token_validator = TokenValidator(
            model=MODEL, base_prompt=guard.base_prompt, max_tokens_threshold=MAX_TOKENS
        )

        split_input = self.token_validator.process(input=self.transaction_string)
        print(len(split_input), split_input)

        parsed_transactions = []
        for input in split_input:
            string, token_number = input

            validated_response = self.get_guard_chat_completion(
                input=string, guard=guard, token_number=token_number
            )

            if validated_response is None:
                raise ValueError("LLM answer does not evaluate into valid object")

            for transaction in validated_response["transaction_list"]:
                parsed_transactions.append(transaction["transaction_info"])

        self.parsed_transactions = parsed_transactions

        return parsed_transactions

    def categorize_transactions(self, transaction_description: str):
        template = """
        You are an expert in assigning categories to a transaction. 
        Be very brief and try to use the least amount of words to the categories you create as possible.
        It's very important that if you can't identify a category you assign "N/A" to it.

        An example of how important it's to be brief:

        example_description: Netflix Super PRemium ultra mega plan
        category: Entertainment

        You will be given the transaction description within triple backticks.


        Your transaction is: ```{transaction_description}```

        Category:
        """

        if self.llm is None:
            self.start_llm()

        # llm_chain = LLMChain(
        #     llm=self.llm, prompt=PromptTemplate.from_template(template)
        # )

        response = self.get_chat_completion(
            prompt=template.format(transaction_description=transaction_description)
        )

        return response.strip()

    def generate_transaction_categories(self):
        """This function will get transactions that appears more than once and assign all of them the same label"""

        # getting unique descriptions
        unique_descriptions = set()
        for transactions in self.parsed_transactions:
            unique_descriptions.add(transactions["transaction_description"])

        transaction_categories = []
        for description in unique_descriptions:
            transaction_categories.append(
                {
                    "transaction_description": description,
                    "transaction_category": self.categorize_transactions(description),
                }
            )

        self.transaction_categories = transaction_categories

        return transaction_categories

    def update_transaction_categories(self, transaction_categories):
        self.transaction_categories = transaction_categories

    def add_categories(self):
        if self.parsed_transactions is None:
            self.parse_transactions()

        if self.transaction_categories is None:
            raise ValueError(
                "Transaction categories are not defined : please run generate_transaction_categories()"
            )

        categorized_transactions = []
        for transaction in self.parsed_transactions:
            # filtering possible obj format for transactions due to LLM randomnes
            if (
                transaction.get("transaction_info", {}).get("transaction_description")
                is not None
            ):
                """format {transaction_info: {transaction_description: 'description'}, ...}"""
                new_transacion_obj = {**transaction["transaction_info"]}
            # Check for format B
            elif transaction.get("transaction_description") is not None:
                """format {transaction_description: 'description', ...}"""
                new_transacion_obj = {**transaction}
            else:
                raise ValueError("Invalid transaction format")

            # create a mapping for self.transaction_categories

            # getting categories
            categories = {
                obj["transaction_description"]: obj["transaction_category"]
                for obj in self.transaction_categories
            }

            new_transacion_obj["category"] = categories[
                new_transacion_obj["transaction_description"]
            ]

            # getting transaction names
            tnames = {
                obj["transaction_description"]: obj["transaction_name"]
                for obj in self.transaction_categories
            }

            new_transacion_obj["transaction_name"] = tnames[
                new_transacion_obj["transaction_description"]
            ]

            categorized_transactions.append(new_transacion_obj)

        return categorized_transactions

    def proprocess_transaction_names(self, text: str):
        # Remove special characters
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Convert to lowercase
        text = text.lower().strip()

        return text

    def clusterize_transactions(
        self, transaction_dt: pd.DataFrame, transaction_description_col: str
    ):
        data = [
            self.proprocess_transaction_names(text)
            for text in transaction_dt[transaction_description_col]
        ]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data)

        # Similarity measurement
        similarity_matrix = linear_kernel(
            X
        )  # this is more efficiente than cosine_similarity function

        # clustering
        Z = linkage(similarity_matrix, "ward")
        clusters = fcluster(Z, t=1.0, criterion="distance")

        transaction_dt["cluster"] = clusters

        return transaction_dt

    def get_cluster_name(self, trasaction_descriptions: list):
        prompt = """
            You are a financial transactions parser specialist. You will be given some financial transactions descriptions within ``` that belong to a same cluster.  
            Provide a shorter and concise name for this cluster

            ``` {transactions}```

            Provide your answer as a string. Don't use the word cluster or any thing related to that. 
            Be as short as you can using as few words possible without losing the essence of these descriptions. 
            If you identify that they all belong to a same company, use the company name.

            Cluster name:
        """

        response = self.get_chat_completion(
            prompt=prompt.format(transactions="\n".join(trasaction_descriptions))
        )

        return response.strip()

    def get_transaction_names(
        self, new_transactions: pd.DataFrame, older_transactions: pd.DataFrame
    ):
        # preparing the data
        cols_of_interest = ["transaction_description", "transaction_name"]

        if older_transactions.empty == True or older_transactions is None:
            all_transactions = new_transactions.copy()
            all_transactions["transaction_name"] = None
        else:
            all_transactions = pd.concat(
                [
                    new_transactions,
                    older_transactions.loc[:, cols_of_interest],
                ]
            )

        all_transactions = self.clusterize_transactions(
            all_transactions, "transaction_description"
        )

        # if any of the transactions in the cluster has a transaction_name
        for cluster in all_transactions["cluster"].unique():
            cluster_transactions = all_transactions[
                all_transactions["cluster"] == cluster
            ]
            # in case there are transaction_names for that same cluster on database
            if cluster_transactions["transaction_name"].notnull().any():
                # get transaction_name
                transaction_name = (
                    cluster_transactions["transaction_name"].dropna().unique()[0]
                )
                # update all transactions in the cluster with that transaction_name
                all_transactions.loc[
                    all_transactions["cluster"] == cluster, "transaction_name"
                ] = transaction_name
            else:
                # call LLM to get cluster name suggestion
                transaction_name = self.get_cluster_name(
                    trasaction_descriptions=cluster_transactions[
                        "transaction_description"
                    ]
                    .unique()
                    .tolist()
                )

                all_transactions.loc[
                    all_transactions["cluster"] == cluster, "transaction_name"
                ] = transaction_name

            # merging the data into the new_transactions because we only want to display the suggestion for the new transactions

        if "transaction_name" in new_transactions.columns:
            new_transactions.drop(columns=["transaction_name"], inplace=True)

        new_transactions = new_transactions.merge(
            all_transactions.loc[:, ["transaction_description", "transaction_name"]]
            .drop_duplicates("transaction_description")
            .reset_index(),
            on="transaction_description",
            how="left",
        )

        return new_transactions
