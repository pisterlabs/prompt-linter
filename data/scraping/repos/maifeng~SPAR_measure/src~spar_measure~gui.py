import json
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from operator import itemgetter
from pathlib import Path

import fire
import gradio as gr
import numpy as np
import openai
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.linalg import inv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoModel, AutoTokenizer

from . import util_funcs
from .zca import ZCA

device = "cuda" if torch.cuda.is_available() else "cpu"


class CVFDemo:
    """
    Demonstrates the Competing Values Frameworks (CVF) measurement using predefined dimensions and scales.

    This class is used as an example in the accompanying paper, showcasing the application of CVF in measuring theoretical concepts.

    Attributes:
        dim_names (list of str): Names of the dimensions used in the CVF model. Example dimensions include 'Create', 'Collaborate', 'Control', and 'Compete'.
        dim_seeds (list of str): Seed sentences for each dimension that define the theoretical concepts. These sentences are used as a starting point for semantic measurement.
        scales (dict): A dictionary defining the scales based on the dimensions. Each scale is a combination of positive and negative aspects of selected dimensions.

    Example scales include 'External-Internal' with positive aspects as 'Create' and 'Compete', and negative aspects as 'Control' and 'Collaborate'; 'Flexible-Stable' with positive aspects as 'Collaborate' and 'Create', and negative aspects as 'Control' and 'Compete'.
    """

    # example values used in the paper
    dim_names = ["Create", "Collaborate", "Control", "Compete"]
    dim_seeds = [
        "We should adapt and innovate.",
        "We should empathize and collaborate.",
        "We should control and stabilize.",
        "We should respond swiftly and serve customers.",
    ]
    scales = {
        "External-Internal": {
            "Positive": ["Create", "Compete"],
            "Negative": ["Control", "Collaborate"],
        },
        "Flexible-Stable": {
            "Positive": ["Collaborate", "Create"],
            "Negative": ["Control", "Compete"],
        },
    }


class PathManager:
    """
    Manages file paths for the project, including output and sample data directories.

    Attributes:
        out_dir (str): Path to the directory where output files will be stored.
        root_dir (str): Path to the root directory of the project.
        sample_data_dir (Path): Path to the directory containing sample data.

    Args:
        out_dir (str): Path to the output directory where files will be saved.

    Raises:
        OSError: If there are issues in determining the root directory or accessing the sample data directory.
    """

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.sample_data_dir = Path(self.root_dir, "sample_data")
        print(self.sample_data_dir)


class Meaurement:
    def __init__(self, path_mgt):
        self.path_mgt = path_mgt

    def read_csv_cols(self, file_obj, measurement_state):
        """
        Reads the columns from a CSV file and updates the measurement state with these columns.

        This method loads a CSV file into a DataFrame and extracts its column names. It then returns two interactive Gradio dropdown components for selecting columns from the loaded DataFrame.

        Args:
            file_obj (file object): The file object of the CSV file to be read.
            measurement_state (dict): A dictionary to store and update the state and data of the measurement process, including the loaded DataFrame and its column names.

        Returns:
            tuple: A pair of Gradio dropdown components for selecting columns from the loaded DataFrame. The first dropdown is for selecting the main column of interest, and the second is for additional selection, both populated with the column names from the CSV file.
        """
        if file_obj is not None:
            measurement_state["input_df"] = pd.read_csv(
                file_obj.name, encoding="ISO-8859-1"
            )
            measurement_state["col_names"] = measurement_state[
                "input_df"
            ].columns.tolist()
            print(f"Columns: {measurement_state['col_names']}")
        return gr.Dropdown(
            choices=measurement_state["col_names"], interactive=True
        ), gr.Dropdown(choices=measurement_state["col_names"], interactive=True)

    def read_input_embedding(self, file_obj, measurement_state):
        """
        Reads and validates the input embedding matrix from a file, updating the measurement state.

        This method loads a NumPy array from the specified file and checks if the number of rows in the embedding matrix matches the number of rows in the input data. It provides feedback through a Gradio textbox and an invisible Gradio button.

        Args:
            file_obj (file object): The file object representing the .npy file containing precomputed embeddings.
            measurement_state (dict): A dictionary used to store the state and data of the measurement process, including the embeddings and input DataFrame.

        Returns:
            tuple: A Gradio textbox with a message indicating the status of the embedding upload, and a Gradio button that is not visible.
        """
        measurement_state["embeddings"] = np.load(file_obj.name)
        if measurement_state["embeddings"].shape[0] != len(
            measurement_state["input_df"]
        ):
            msg = f"Error: The number of rows in the input data and the number of rows in the embedding matrix do not match! The shape of the embeddings is {measurement_state['embeddings'].shape}, and the number of rows in the input data is {len(measurement_state['input_df'])}. "
        else:
            msg = f"Precomputed embedding uploaded! \nThe shape of the embeddings is {measurement_state['embeddings'].shape}. Proceed to the next tab to define the dimensions."
        return gr.Textbox(value=msg), gr.Button(visible=False)

    def set_doc_col(self, doc_col_name, doc_id_col_name, measurement_state):
        """
        Sets the document and document ID column names in the measurement state.

        This method configures the measurement state with the specified document and document ID column names, ensuring that these columns exist in the input DataFrame. It returns a Gradio textbox to indicate the successful setting of column names.

        Args:
            doc_col_name (str): The name of the column in the input DataFrame containing the document text.
            doc_id_col_name (str): The name of the column in the input DataFrame containing the document IDs.
            measurement_state (dict): A dictionary used to store and update the state and data of the measurement process, including the names of the document and document ID columns.

        Returns:
            gr.Textbox: A Gradio textbox indicating that the column names have been successfully set in the measurement state.
        """
        print(f"Setting doc_col_name to {doc_col_name}")
        if measurement_state["input_df"].columns.tolist() is not None:
            if doc_col_name in measurement_state["input_df"].columns.tolist():
                measurement_state["doc_col_name"] = doc_col_name

        print(f"Setting doc_id_col_name to {doc_id_col_name}")
        if measurement_state["input_df"].columns.tolist() is not None:
            if doc_id_col_name in measurement_state["input_df"].columns.tolist():
                measurement_state["doc_id_col_name"] = doc_id_col_name
        return gr.Textbox(value="Column names set!")

    def set_sbert(self, model_name, measurement_state):
        """
        Configures the Sentence-BERT (SBERT) model for text embedding.

        This method sets up the SBERT model specified by the model_name. It updates the measurement state with the model name, tokenizer, and the pre-trained model itself. The model is set to evaluation mode and moved to the appropriate computing device.

        Args:
            model_name (str): The name of the pre-trained Sentence-BERT model to be used.
            measurement_state (dict): A dictionary used to store the state and data of the measurement process, including the SBERT model and tokenizer.

        Returns:
            None: This method does not return anything but updates the measurement_state with the SBERT model and tokenizer.
        """
        print(f"Setting SBERT model to {model_name}")
        measurement_state["model_name"] = model_name
        measurement_state["tokenizer"] = AutoTokenizer.from_pretrained(
            f"sentence-transformers/{model_name}"
        )
        measurement_state["model"] = AutoModel.from_pretrained(
            f"sentence-transformers/{model_name}"
        )
        measurement_state["model"].eval()
        measurement_state["model"].to(device)

    def set_openai_api_key(self, api_key, measurement_state):
        """
        Sets the OpenAI API key in the measurement state for accessing OpenAI services.

        This method attempts to set the OpenAI API key either from the provided api_key argument or from the environment variable 'OPENAI_API_KEY'. If successful, it updates the measurement_state with the API key; otherwise, it returns a Gradio textbox indicating an error.

        Args:
            api_key (str): The OpenAI API key provided by the user. If empty, the method will attempt to use the API key from the environment variable.
            measurement_state (dict): A dictionary used to store the state and data of the measurement process, including the OpenAI API key.

        Returns:
            gr.Textbox (optional): A Gradio textbox that is visible only if there is an error in setting the API key, displaying an appropriate error message.
        """
        print("Setting OpenAI API key")
        if api_key != "":
            try:
                measurement_state["openai_api_key"] = api_key
                print("Using API key from textbox.")
            except Exception as e:
                print(e)
                return gr.Textbox(
                    value="⚠️ Invalid API key. Please try again.", visible=True
                )
        else:
            print("Using environment variable (OPENAI_API_KEY) for API key.")
            try:
                measurement_state["openai_api_key"] = os.environ["OPENAI_API_KEY"]
            except Exception as e:
                return gr.Textbox(
                    value="⚠️ Invalid API key in OPENAI_API_KEY environment variable. Please try again.",
                    visible=True,
                )

    def toggle_embedding_model_visibility(self, embedding_model_dropdown):
        """
        Toggles the visibility of different embedding model options based on the selected embedding model.

        This method adjusts the visibility of Gradio components (textboxes and a button) based on whether the user selects a 'Sentence Transformers' model or an 'OpenAI' model in the dropdown.
        Args:
            embedding_model_dropdown (str): The current value of the embedding model dropdown, indicating the selected embedding model type.

        Returns:
            tuple: A tuple of three Gradio components:
                - A Gradio textbox for Sentence Transformers, visible if a Sentence Transformers model is selected.
                - A Gradio textbox for OpenAI, visible if an OpenAI model is selected.
                - A Gradio button to set the embedding model.
        """
        if embedding_model_dropdown.startswith("Sentence Transformers"):
            return (
                gr.Textbox(visible=True),
                gr.Textbox(visible=False),
                gr.Button(value="Set Embedding Model"),
            )
        if embedding_model_dropdown.startswith("OpenAI"):
            return (
                gr.Textbox(visible=False),
                gr.Textbox(visible=True),
                gr.Button(value="Set Embedding Model"),
            )

    def set_embedding_model(
        self,
        embedding_model_dropdown,
        sbert_model_textbox,
        openai_api_key,
        measurement_state,
    ):
        """
        Sets the embedding model in the measurement state based on the user's selection.

        This method configures the measurement state with the selected embedding model, either a Sentence Transformers model or using the OpenAI API. It also updates the measurement state to indicate which embedding service is in use.

        Args:
            embedding_model_dropdown (str): The selected embedding model type from the dropdown.
            sbert_model_textbox (str): The name of the Sentence-BERT model to be used, applicable if Sentence Transformers is selected.
            openai_api_key (str): The OpenAI API key, applicable if OpenAI is selected.
            measurement_state (dict): A dictionary used to store the state and data of the measurement process.

        Returns:
            gr.Button: A Gradio button indicating that the embedding model has been set.
        """
        print(f"Setting embedding model to {embedding_model_dropdown}")
        if embedding_model_dropdown.startswith("Sentence Transformers"):
            self.set_sbert(sbert_model_textbox, measurement_state)
            measurement_state["use_openAI"] = False
        if embedding_model_dropdown.startswith("OpenAI"):
            self.set_openai_api_key(openai_api_key, measurement_state)
            measurement_state["use_openAI"] = True
        return gr.Button(value="Embedding model set!")

    def reset_set_emb_btn(self):
        return gr.Button(value="Set Embedding Model")

    def set_embedding_option(self, embedding_option):
        """
        Adjusts the visibility of Gradio interface components based on the selected embedding option.

        This method dynamically updates the user interface to show relevant components for either uploading precomputed embeddings or embedding documents within the app. It returns a Gradio file input and a Gradio button with adjusted visibility.

        Args:
            embedding_option (str): The selected option for handling embeddings, either 'Upload Precomputed Embeddings' or 'Embed Documents'.

        Returns:
            tuple: A pair of Gradio components:
                - A Gradio file input, visible for uploading precomputed embeddings and hidden for embedding documents within the app.
                - A Gradio button, hidden for uploading precomputed embeddings and visible for embedding documents within the app.
        """
        if embedding_option == "Upload Precomputed Embeddings":
            return gr.File(visible=True), gr.Button(visible=False)
        if embedding_option == "Embed Documents":
            return gr.File(visible=False), gr.Button(visible=True)

    @classmethod
    def embed_texts(cls, docs, progress, measurement_state):
        """
        Embeds a list of docs using the specified embedding model in the measurement state.

        This method takes a list of sentences and embeds them using either Sentence Transformers or OpenAI, based on the configuration in the measurement state. It handles the embedding process in batches for Sentence Transformers and individually for OpenAI.

        Args:
            docs (list of str): A list of docs to be embedded.
            progress (tqdm): An instance of the tqdm class for displaying progress.
            measurement_state (dict): A dictionary containing the state and data of the measurement process, including the chosen embedding model and relevant configurations.

        Returns:
            np.ndarray: An array of embedded docs. The shape of the array depends on the number of docs and the embedding model used.

        Raises:
            AssertionError: If the number of embeddings does not match the number of input docs when using Sentence Transformers.
            openai.error.InvalidRequestError: If there is an error in embedding using the OpenAI API, such as invalid API key or request format.
        """
        batch_size = 8
        docs = [str(x) for x in docs]
        if measurement_state["use_openAI"] is False:
            # use sentence_transformers to embed the text_col
            print("Embedding Using Sentence Transformers")
            # use batching
            with torch.no_grad():
                measurement_state["model"].eval()
                sentence_embeddings = []
                for i in progress.tqdm(
                    range(0, len(docs), batch_size), unit="batch (batch size=8)"
                ):
                    batch = docs[i : i + batch_size]
                    encoded_input = measurement_state["tokenizer"](
                        batch, padding=True, truncation=True, return_tensors="pt"
                    ).to(device)
                    model_output = measurement_state["model"](**encoded_input)
                    batch_sentence_embeddings = util_funcs.mean_pooling(
                        model_output, encoded_input["attention_mask"]
                    )
                    batch_sentence_embeddings = F.normalize(
                        batch_sentence_embeddings, p=2, dim=1
                    )
                    sentence_embeddings.append(batch_sentence_embeddings)
                sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
                # to cpu and numpy
                sentence_embeddings = sentence_embeddings.cpu().detach().numpy()
                print(f"Shape of sentence embeddings: {sentence_embeddings.shape}")
                assert sentence_embeddings.shape[0] == len(docs)

        else:
            print("Embedding Using OpenAI")

            @retry(
                stop=stop_after_attempt(20),
                wait=wait_exponential(
                    multiplier=1, min=2, max=60
                ),  # use exponential backoff with a minimum wait of 4 seconds and a maximum of 10
                after=util_funcs.print_error,
            )
            def call_openai_api(batch, model):
                client = OpenAI(api_key=measurement_state["openai_api_key"])
                return client.embeddings.create(input=batch, model=model)

            # Function to divide the docs into batches of 8
            def divide_into_batches(docs, batch_size):
                for i in range(0, len(docs), batch_size):
                    yield docs[i : i + batch_size]

            # Batch the docs
            sentence_batches = list(divide_into_batches(docs, batch_size=8))

            # List to store all embeddings
            sentence_embeddings = []

            # Process each batch
            for batch in progress.tqdm(sentence_batches, unit="batch"):
                try:
                    batch_response = call_openai_api(batch, "text-embedding-ada-002")
                    embeddings = [
                        response.embedding for response in batch_response.data
                    ]
                    sentence_embeddings.extend(embeddings)
                except openai.error.InvalidRequestError as e:
                    print(f"An error occurred: {e}")
                    # Handle the error as needed

            # Convert to numpy array and normalize
            sentence_embeddings = np.array(sentence_embeddings)
            sentence_embeddings = sentence_embeddings / np.linalg.norm(
                sentence_embeddings, axis=1, keepdims=True
            )
            print(f"Shape of sentence embeddings: {sentence_embeddings.shape}")

        return sentence_embeddings

    def embed_df(self, measurement_state, progress=gr.Progress()):
        """
        Embeds the documents from the input DataFrame and saves the embeddings to a file.

        This method embeds the text from a specified column in the input DataFrame using the configured embedding model. It saves the embeddings as a .npy file and provides Gradio components for displaying the embedding status and downloading the embeddings.

        Args:
            measurement_state (dict): A dictionary containing the state and data of the measurement process, including the input DataFrame and the name of the column to be embedded.
            progress (gr.Progress, optional): A Gradio progress component for tracking the embedding process.

        Returns:
            tuple: A tuple containing:
                - A Gradio textbox with a message indicating the status of the embedding process.
                - A Gradio file component for downloading the embeddings file.
                - A string representing the file path of the saved embeddings.

        Raises:
            Exception: If there is an error during the embedding process, with an error message included in the Gradio textbox.
        """
        try:
            measurement_state["embeddings"] = self.embed_texts(
                measurement_state["input_df"][measurement_state["doc_col_name"]],
                progress=progress,
                measurement_state=measurement_state,
            )
            np.save(
                Path(self.path_mgt.out_dir, "embeddings.npy"),
                measurement_state["embeddings"],
            )
            if (measurement_state["doc_col_name"] is not None) and (
                Path(self.path_mgt.out_dir, "embeddings.npy").exists()
            ):
                return (
                    gr.Textbox(
                        value=f"Embedding completed! \nThe shape of the embeddings is {measurement_state['embeddings'].shape}. You can download and save the embeddings below. Proceed to the next tab to define the dimensions."
                    ),
                    gr.File(visible=True),
                    str(Path(self.path_mgt.out_dir, "embeddings.npy")),
                )
        except Exception as e:
            return (
                gr.Textbox(
                    value=f"⚠️ Embedding failed! Make sure that you have clicked Confirm Column Selections and Set Embedding Model. \nError message: {e}"
                ),
                gr.File(visible=False),
                None,
            )

    def toggle_row_vis(self, n_rows, measurement_state):
        """
        Toggles the visibility of rows in the Gradio interface based on the number of dimensions specified.

        This method dynamically updates the visibility of rows in the Gradio interface on the dimensions tab. It sets the number of visible rows according to the specified number of dimensions and updates the measurement state accordingly.

        Args:
            n_rows (int): The number of rows (dimensions) to be made visible in the Gradio interface.
            measurement_state (dict): A dictionary used to store the state and data of the measurement process, including the number of dimensions.

        Returns:
            list of gr.Row: A list of Gradio Row objects with updated visibility. The first n_rows are set to visible, and the rest are set to invisible.

        Raises:
            ValueError: If n_rows is not an integer or is outside the acceptable range (e.g., negative numbers or numbers larger than the total number of available rows).
        """
        # toggle visbility of rows (dimensions tab)
        update_rows = []
        measurement_state["n_dims"] = n_rows
        for i in range(10):
            if i < n_rows:
                update_rows.append(gr.Row(visible=True))
            else:
                update_rows.append(gr.Row(visible=False))
        return update_rows

    def toggle_row_vis_scales(self, n_rows, measurement_state):
        """
        Adjusts the visibility of rows in the Gradio interface on the scales tab based on the specified number of scales.

        This method updates the number of visible rows in the Gradio interface on the scales tab according to the number of scales defined by the user. It alters the measurement state to reflect the new number of scales.

        Args:
            n_rows (int): The number of rows (scales) to be made visible on the scales tab.
            measurement_state (dict): A dictionary used to store the state and data of the measurement process, including the number of scales.

        Returns:
            list of gr.Row: A list of Gradio Row objects with visibility updated according to the specified number of scales. Rows corresponding to the defined number of scales are made visible, while the rest are hidden.

        Raises:
            ValueError: If n_rows is not a valid integer or if it exceeds the predefined maximum number of rows.
        """
        # toggle visbility of rows (scales tab)
        update_rows = []
        measurement_state["n_scales"] = n_rows
        for i in range(10):
            if i < n_rows:
                update_rows.append(gr.Row(visible=True))
            else:
                update_rows.append(gr.Row(visible=False))
        return update_rows

    def semantic_search(
        self, query, n_results, measurement_state, progress=gr.Progress()
    ):
        """
        Performs semantic search to find the most similar documents to the provided query.

        This method takes a query, splits it into individual lines, and uses the configured embedding model to embed each line, take the average embedding, and find the most similar documents in the input data. The results are displayed in a Gradio textbox, showing the document IDs, scores, and excerpts of the top matches.

        Args:
            query (str): A string containing one or more queries, separated by new lines.
            n_results (int): The number of top results to return from the semantic search.
            measurement_state (dict): A dictionary containing the state and data of the measurement process, including the document embeddings.
            progress (gr.Progress, optional): A Gradio progress component for tracking the embedding process of the query.

        Returns:
            gr.Textbox: A Gradio textbox containing the results of the semantic search, including document IDs, similarity scores, and excerpts.

        Raises:
            ValueError: If the query is empty or invalid.
            AssertionError: If there is a mismatch in the length of the lists of document IDs, scores, and excerpts in the search results.
            Exception: If there is an error during the embedding or search process, with the error message included in the Gradio textbox.
        """
        result_str = ""
        queries = query.split("\n")
        # remove empty queries or queries with only spaces
        queries = [q for q in queries if q.strip() != ""]
        if (len(queries) == 0) or (queries is None) or all(q == "" for q in queries):
            return gr.Textbox(value="No query provided!")
        try:
            query_embedding = self.embed_texts(
                queries, progress=progress, measurement_state=measurement_state
            )
            mean_vect_query = query_embedding.mean(axis=0)
            hits = util_funcs.semantic_search(
                mean_vect_query,
                measurement_state["embeddings"],
                score_function=util_funcs.dot_score,
                top_k=n_results,
            )
            hit_ids = [h["corpus_id"] for h in hits[0]]
            hit_scores = [h["score"] for h in hits[0]]
            examples = itemgetter(*hit_ids)(
                measurement_state["input_df"][measurement_state["doc_col_name"]]
            )
            examples = list(examples)
            example_doc_ids = itemgetter(*hit_ids)(
                measurement_state["input_df"][measurement_state["doc_id_col_name"]]
            )
            example_doc_ids = list(example_doc_ids)
            assert len(hit_ids) == len(hit_scores) == len(examples)
            for i in range(len(hit_ids)):
                result_str += f"Document ID: {example_doc_ids[i]} \n"
                result_str += f"Score: {round(hit_scores[i], 3)} \n"
                result_str += "------------------------ \n"
                result_str += f"{examples[i]}\n"
                result_str += "------------------------ \n"
        except Exception as e:
            result_str = (
                f"⚠️ Error. Make sure the query is not empty. \nError message: {e}"
            )
        return gr.Textbox(value=result_str)

    def save_dims(self, measurement_state, progress=gr.Progress(), *dims_boxes):
        """
        Saves the embeddings and queries for user-defined dimensions, updating the measurement state.

        This method processes user-input dimensions, embeds their queries, and stores the embeddings and queries in the measurement state. It also saves the dimension queries as a JSON file and returns Gradio components to reflect the saved dimensions and allow for the download of the JSON file.

        Args:
            measurement_state (dict): A dictionary containing the state and data of the measurement process.
            progress (gr.Progress, optional): A Gradio progress component for tracking the embedding process of the dimension queries.
            *dims_boxes (str): Variable length argument list containing dimension names and queries.

        Returns:
            list: A list containing Gradio components. It includes Dropdown components for scale selection, a Textbox with a message about the saved dimensions, a File component for downloading the dimension queries, and a string representing the file path of the JSON file.

        Raises:
            Exception: If there is an error during processing or saving the dimensions, with the error message included in the Gradio textbox.
        """
        try:
            # first half of dims_boxes are dim names, second half are dim queries
            all_dim_names = dims_boxes[: len(dims_boxes) // 2]
            all_dim_queries = dims_boxes[len(dims_boxes) // 2 :]
            measurement_state["dim_embeddings"] = {}
            measurement_state["dim_queries"] = {}
            for i in range(measurement_state["n_dims"]):
                if all_dim_names[i].strip() != "":
                    dim_name = all_dim_names[i].strip()
                else:  # if no name is given, use default name
                    dim_name = f"Dimension_{i+1}"
                dim_queries = all_dim_queries[i].split("\n")  # dim queries
                # remove empty queries or queries with only spaces
                dim_queries = [q for q in dim_queries if q.strip() != ""]
                dim_embedding = self.embed_texts(
                    dim_queries, progress=progress, measurement_state=measurement_state
                )
                mean_vect_dim = dim_embedding.mean(axis=0)
                measurement_state["dim_embeddings"][dim_name] = mean_vect_dim
                measurement_state["dim_queries"][dim_name] = dim_queries
            # save measurement_state['dim_queries'] to json file
            with open(Path(self.path_mgt.out_dir, "dimensions_queries.json"), "w") as f:
                json.dump(measurement_state["dim_queries"], f)
            # update both positive and negative scale selectors
            return (
                [gr.Dropdown(choices=list(measurement_state["dim_queries"].keys()))]
                * 20
                + [
                    gr.Textbox(
                        visible=True,
                        value=f"Dimensions saved: {list(measurement_state['dim_queries'].keys())}. You can download the json file below to keep a record of the final queries. Proceed to the next tab to define scales.",
                    )
                ]
                + [gr.File(visible=True)]
                + [str(Path(self.path_mgt.out_dir, "dimensions_queries.json"))]
                + [gr.Markdown(visible=False)]
            )
        except Exception as e:
            return (
                [gr.Dropdown(choices=list(measurement_state["dim_queries"].keys()))]
                * 20
                + [
                    gr.Textbox(
                        visible=True,
                        value=f"⚠️ Please check that you have filled in all the dimension names and queries. You can move the slider 'Number of dimensions' on the top to add or remove dimensions. \n  Error message: {e}.",
                    )
                ]
                + [gr.File(visible=False)]
                + [None]
                + [gr.Markdown(visible=True)]
            )

    def save_scales(self, measurement_state, *scale_boxes):
        """
        Saves the embeddings and definitions for user-defined scales, updating the measurement state.

        This method processes user-input scales, computes their embeddings by averaging the embeddings of the positive and negative dimensions, and stores the scale embeddings and definitions in the measurement state. It also saves the scale definitions as a JSON file and returns Gradio components to reflect the saved scales and allow for the download of the JSON file.

        Args:
            measurement_state (dict): A dictionary containing the state and data of the measurement process.
            *scale_boxes (str): Variable length argument list containing scale names, positive dimensions, and negative dimensions.

        Returns:
            list: A list containing Gradio components. It includes a Textbox with a message about the saved scales, a File component for downloading the scale definitions, and a string representing the file path of the JSON file.

        Raises:
            Exception: If there is an error during processing or saving the scales, with the error message included in the Gradio textbox. This includes cases where scale names or dimensions are missing or improperly defined.
        """
        try:
            # first 1/3 of scales boxes are names, second 1/3 are positive scales, third 1/3 are negative scales
            all_scale_names = scale_boxes[: len(scale_boxes) // 3]
            all_pos_scales = scale_boxes[
                len(scale_boxes) // 3 : 2 * len(scale_boxes) // 3
            ]
            all_neg_scales = scale_boxes[2 * len(scale_boxes) // 3 :]
            measurement_state["scale_embeddings"] = {}
            measurement_state["scale_definitions"] = {}
            # all scale embeddings are average of positive scales subtracted by average of negative scales
            for i in range(measurement_state["n_scales"]):
                if all_scale_names[i].strip() != "":
                    scale_name = all_scale_names[i].strip()
                else:  # if no name is given, use default name
                    scale_name = f"Scale_{i+1}"
                # save scale definitions
                """
                example scale_definitions json format: 
                scales = {
                    "External-Internal": {
                        "Positive": ["Create", "Compete"],
                        "Negative": ["Control", "Collaborate"],
                    },
                    "Flexible-Stable": {
                        "Positive": ["Collaborate", "Create"],
                        "Negative": ["Control", "Compete"],
                    },
                }
                """
                measurement_state["scale_definitions"][scale_name] = {
                    "Positive": all_pos_scales[i],
                    "Negative": all_neg_scales[i],
                }
                scale_embedding_pos = []
                scale_embedding_neg = []
                if len(all_pos_scales[i]) > 0:
                    for pos_scale in all_pos_scales[i]:
                        scale_embedding_pos.append(
                            measurement_state["dim_embeddings"][pos_scale]
                        )
                if len(all_neg_scales[i]) > 0:
                    for neg_scale in all_neg_scales[i]:
                        scale_embedding_neg.append(
                            measurement_state["dim_embeddings"][neg_scale]
                        )
                if (len(scale_embedding_pos) > 0) & (len(scale_embedding_neg) > 0):
                    # take averages and difference
                    scale_embedding = np.stack(scale_embedding_pos).mean(
                        axis=0
                    ) - np.stack(scale_embedding_neg).mean(axis=0)
                elif len(scale_embedding_pos) > 0:  # only positive scales
                    scale_embedding = np.stack(scale_embedding_pos).mean(axis=0)
                elif len(scale_embedding_neg) > 0:  # only negative scales
                    scale_embedding = -np.stack(scale_embedding_neg).mean(axis=0)
                else:  # no scales
                    # throw exception
                    raise Exception(
                        f"Check that you have filled in all the scale names and at least one positive or negative scale for scale {scale_name}."
                    )
                measurement_state["scale_embeddings"][scale_name] = scale_embedding

            # save measurement_state['scales'] to json file
            with open(Path(self.path_mgt.out_dir, "scale_definitions.json"), "w") as f:
                json.dump(measurement_state["scale_definitions"], f)

            return (
                [
                    gr.Textbox(
                        visible=True,
                        value=f"Scales saved: {list(measurement_state['scale_definitions'].keys())}. You can download the json file below to keep a record of the scale definitions. Proceed to the next tab to measure using semantic projection.",
                    )
                ]
                + [gr.File(visible=True)]
                + [str(Path(self.path_mgt.out_dir, "scale_definitions.json"))]
                + [gr.Markdown(visible=False)]
            )
        except Exception as e:
            # return error message in textbox update
            return (
                [
                    gr.Textbox(
                        visible=True,
                        value=f"⚠️ Error: Please check your scale definitions. Make sure that you have clicked the Embed Queries and Save Dimensions button in Tab 2, and there is no empty scales or dimensions. You can move the slider 'Number of scales' on the top to add or remove scales. \nError message: {e}",
                    )
                ]
                + [gr.File(visible=False)]
                + [None]
                + [gr.Markdown(visible=True)]
            )

    def measure_docs(self, single_subspace: str, whitening: str, measurement_state):
        """
        Measures documents against defined scales and outputs the results as a CSV file.

        This method computes the similarity of document embeddings to scale embeddings using the dot product or a subspace projection method. Optionally, it applies whitening transformation to the measures. The results are saved as a CSV file, and Gradio components are returned for displaying the completion message and downloading the results.

        Args:
            single_subspace (str): A flag indicating whether to use a general subspace projection that treats all scales a single subspace (think of it as measuring one scale while controlling for other scales). Expected values are "Yes" or "No".
            whitening (str): A flag indicating whether to apply whitening transformation to the measures. Expected values are "Yes" or "No".
            measurement_state (dict): A dictionary containing the state and data of the measurement process, including document and scale embeddings.

        Returns:
            tuple: A tuple containing:
                - A Gradio textbox with a message indicating the completion of the measurement process.
                - A Gradio file component for downloading the measurement results.
                - A string representing the file path of the saved measurement results.

        Raises:
            ValueError: If the values for single_subspace or whitening are not recognized.
            FileNotFoundError: If there is an error in saving the measurement results to a CSV file.
            numpy.linalg.LinAlgError: If there is an error in matrix inversion during subspace projection.
        """
        print(
            f"Measuring with the following arguments: single_subspace={single_subspace}, whitening={whitening}."
        )
        scales_embs = []
        for scale in list(measurement_state["scale_definitions"].keys()):
            scales_embs.append(measurement_state["scale_embeddings"][scale])
        scales_embs = np.stack(scales_embs).T

        # normalize scales_embs
        scales_embs = scales_embs / np.linalg.norm(scales_embs, axis=0, keepdims=True)
        scales_embs = scales_embs.T
        if single_subspace == "No":
            measures = (
                util_funcs.dot_score(
                    measurement_state["embeddings"],
                    scales_embs,
                )
                .cpu()
                .numpy()
            )
        else:
            # measures = [(S'S)^-1]S'X where S is the scale embeddings and X is the document embeddings
            measures = (
                inv((scales_embs).dot(scales_embs.T))
                .dot(scales_embs)
                .dot(measurement_state["embeddings"].T)
            )
            measures = measures.T

        if whitening == "Yes":
            # whitening
            trf = ZCA().fit(measures)
            measures = trf.transform(measures)

        # output as csv to File
        scale_measures = pd.DataFrame(measures)
        scale_measures = scale_measures.round(4)
        scale_measures.columns = list(measurement_state["scale_definitions"].keys())
        scale_measures[measurement_state["doc_id_col_name"]] = measurement_state[
            "input_df"
        ][measurement_state["doc_id_col_name"]]
        scale_measures = scale_measures[
            [measurement_state["doc_id_col_name"]]
            + list(measurement_state["scale_definitions"].keys())
        ]
        Path("measure_output").mkdir(parents=True, exist_ok=True)
        Path(self.path_mgt.out_dir, "measurement_output.csv")
        scale_measures.to_csv(
            Path(self.path_mgt.out_dir, "measurement_output.csv"), index=False
        )
        return (
            gr.Textbox(
                visible=True,
                value="Measurement completed. Download the results below. ",
            ),
            gr.File(visible=True),
            str(Path(self.path_mgt.out_dir, "measurement_output.csv")),
        )

    def load_example_dataset(self, measurement_state):
        """
        Loads an example dataset and updates the measurement state with predefined settings.

        This method reads a sample text dataset and its precomputed embeddings, sets the Sentence-BERT model and document-related column names, and updates Gradio interface components to reflect predefined dimensions and scales from the CVFDemo class.

        Args:
            measurement_state (dict): A dictionary to store the state and data of the measurement process, including the loaded example dataset and embeddings.

        Returns:
            list: A list of Gradio components configured with example dataset settings. It includes components for dataset and embedding information, document column selection, embedding model selection, dimension names and seeds, scale names and selectors, and sliders for dimensions and scales.

        Raises:
            FileNotFoundError: If the sample text CSV file or the precomputed embeddings file cannot be found.
            pd.errors.ParserError: If there is an error in parsing the sample text CSV file.
            numpy.lib.npyio.NpyFileError: If there is an error in loading the precomputed embeddings file.
        """
        # tab 1
        measurement_state["input_df"] = pd.read_csv(
            Path(self.path_mgt.sample_data_dir, "sample_text.csv")
        )
        measurement_state["embeddings"] = np.load(
            Path(self.path_mgt.sample_data_dir, "sample_emb.npy")
        )
        self.set_sbert("all-MiniLM-L6-v2", measurement_state)
        measurement_state["doc_col_name"] = "text"
        measurement_state["doc_id_col_name"] = "doc_id"
        measurement_state["use_openAI"] = False

        # tab 2
        all_dim_name_boxes_updates = []
        all_dim_seed_boxes_updates = []
        for i in range(10):
            if i < len(CVFDemo().dim_names):
                all_dim_name_boxes_updates.append(
                    gr.Textbox(value=f"{CVFDemo().dim_names[i]}")
                )
                all_dim_seed_boxes_updates.append(
                    gr.Textbox(value=f"{CVFDemo().dim_seeds[i]}")
                )
            else:
                all_dim_name_boxes_updates.append(gr.Textbox(value=""))
                all_dim_seed_boxes_updates.append(gr.Textbox(value=""))

        # tab 3
        all_scale_name_boxes_updates = []
        all_scale_pos_selector_updates = []
        all_scale_neg_selector_updates = []

        demo_scale_names = list(CVFDemo().scales.keys())
        for i in range(10):
            if i < 2:
                all_scale_name_boxes_updates.append(
                    gr.Textbox(value=demo_scale_names[i])
                )
                all_scale_pos_selector_updates.append(
                    gr.Textbox(value=CVFDemo().scales[demo_scale_names[i]]["Positive"])
                )
                all_scale_neg_selector_updates.append(
                    gr.Textbox(value=CVFDemo().scales[demo_scale_names[i]]["Negative"])
                )
            else:
                all_scale_name_boxes_updates.append(None)
                all_scale_pos_selector_updates.append(None)
                all_scale_neg_selector_updates.append(None)

        return (
            [
                gr.Row(visible=True),
                str(Path(self.path_mgt.sample_data_dir, "sample_text.csv")),
                str(Path(self.path_mgt.sample_data_dir, "sample_emb.npy")),
                gr.Radio(value="Embed Documents"),
                gr.Textbox(
                    value="""Embedding completed! The shape of the embeddings is (2000, 384). You can download and save the embeddings below. Proceed to the next tab to define the dimensions."""
                ),
                gr.Dropdown(value="text"),
                gr.Dropdown(value="doc_id"),
                gr.Dropdown(value="Sentence Transformers (Local)"),
                gr.Textbox(value="all-MiniLM-L6-v2"),
            ]
            + all_dim_name_boxes_updates
            + all_dim_seed_boxes_updates
            + all_scale_name_boxes_updates
            + all_scale_pos_selector_updates
            + all_scale_neg_selector_updates
            + [gr.Slider(value=4), gr.Slider(value=2)]
        )


def run_gui(
    out_dir="measure_output/", mode="local", username=None, password=None, **kwargs
):
    """
    Launches the SPAR Gradio web interface for semantic text measurement.

    This function sets up and runs the Gradio Blocks interface for SPAR, providing tabs for data upload and embedding, dimension and scale definition, and document measurement. It supports both local and public modes, with optional authentication.

    Args:
        out_dir (str, optional): The directory where output files will be saved. Defaults to 'measure_output/'.
        mode (str, optional): The mode of the Gradio interface, either 'local' for running on a local machine or 'public' for public web access. Defaults to 'local'.
        username (str, optional): The username for authentication if running in public mode. No authentication is required if None. Defaults to None.
        password (str, optional): The password for authentication if running in public mode. No authentication is required if None. Defaults to None.
        **kwargs: Additional keyword arguments passed to the Gradio launch function.

    Returns:
        None: This function does not return a value but launches the Gradio web interface.

    Raises:
        ValueError: If the mode is not 'local' or 'public'.
        OSError: If there are issues in creating the output directory or setting the file paths.
        RuntimeError: If there are issues in initializing or running the Gradio interface.
    """

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path_mgt = PathManager(out_dir=out_dir)

    torch.set_grad_enabled(False)  # disable autograd

    with gr.Blocks(title="SPAR") as demo:
        # set up inital measurement state ========================================================
        m = Meaurement(path_mgt=path_mgt)
        state = gr.State({})
        gr.Markdown(
            "### SPAR: Semantic Projection with Active Retrieval (Research Preview)"
        )
        all_dim_name_boxes = []
        all_search_query_boxes = []
        all_search_btns = []
        all_search_results = []
        all_rows_dims = []
        all_rows_scale = []
        all_scale_name_boxes = []
        all_scale_pos_selector = []
        all_scale_neg_selector = []
        # Basic Info ==================================================================================
        with gr.Row():
            with gr.Column(scale=1):
                example_btn = gr.Button(value="💡 Load Example Dataset and Scales")
            with gr.Column(scale=8):
                gr.Markdown(
                    value="""* SPAR is a Python package and a web interface for measuring short text documents using semantic projection.
                    * The package is developed as part of the paper:
                        __Bei Yan, Feng Mai, Chaojiang Wu, Rui Chen, Xiaolin Li (2023). A Computational Framework for Understanding Firm Communication During Disasters. Information Systems Research. https://doi.org/10.1287/isre.2022.0128__  
                        It is a research prototype and under active development. Please contact the authors for any questions or feedback.
                    * The source code is available on [GitHub](https://github.com/maifeng/SPAR_measure) under GPLv3 license.""",
                    label="",
                )
        example_row = gr.Row(visible=False, variant="panel")
        with example_row:
            with gr.Column(scale=2):
                example_load_msgbox = gr.Markdown(
                    visible=True,
                    label="",
                    value="""
                    
                    💡 __Example dataset and embeddings loaded. You may download the example dataset and embeddings on the right.__  
                    💡 __Do not change the settings in Tab 1. Proceed to Tab 2.__  
                    💡 __If you want to use your own dataset, refresh the page and upload a CSV file in Tab 1.__""",
                )
            with gr.Column(scale=1):
                example_file_download = gr.File(
                    visible=True, label="Download Example File"
                )
                example_emb_download = gr.File(
                    visible=True, label="Download Example Embeddings"
                )
        # Tab 1 =========================================================================================================
        with gr.Tab("1. Upload File and Embed"):
            # read csv file and select columns
            gr.Markdown(
                value=" 📖 Upload a **CSV file** that contains at least 2 columns: (1) the documents to be measured, and (2) the document IDs. Alternatively, you can click the Load Example Dataset and Scales button on top to explore the tool with an included sample dataset (2000 Facebook posts) and pre-defined Competing Values Framework (CVF) dimension and scales.",
                label="",
            )
            with gr.Row(variant="panel"):
                input_file = gr.File(
                    file_count="single",
                    file_types=[".csv"],
                    label="Input CSV File",
                )

                # read_col_btn = gr.Button(value="Read CSV File Columns")
                doc_col_selector = gr.Dropdown(
                    choices="",
                    label="Select Text Column",
                    interactive=False,
                    allow_custom_value=True,
                )
                doc_id_col_selector = gr.Dropdown(
                    choices="",
                    label="Select Document ID Column",
                    interactive=False,
                    allow_custom_value=True,
                )
                doc_id_col_btn = gr.Button(value="Confirm Column Selections")
                doc_id_col_btn.click(
                    fn=m.set_doc_col,
                    inputs=[doc_col_selector, doc_id_col_selector, state],
                    outputs=doc_id_col_btn,
                )

            gr.Markdown(
                value=""" 📖 Select an embedding method. You can use Sentence Transformers to embed the text locally; in this case you can use the default model name (all-MiniLM-L6-v2) or [any other models](https://www.sbert.net/docs/pretrained_models.html). Larger models will take longer to embed but may produce better results.
                Alternatively, you can use the [OpenAI API](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) to embed (text-embedding-ada-002 model). You can get an API key [here](https://platform.openai.com/account/api-keys). """,
            )

            with gr.Row(variant="panel"):
                # embedding options
                embedding_model_dropdown = gr.Dropdown(
                    choices=[
                        "Sentence Transformers (Local)",
                        "OpenAI text-embedding-ada-002 (API Key Required)",
                    ],
                    label="Select an embedding method:",
                    interactive=True,
                    value="Sentence Transformers (Local)",
                    allow_custom_value=True,
                    multiselect=False,
                )
                openai_api_key = gr.Textbox(
                    placeholder="",
                    label="OpenAI API Key (leave blank if key is set as Environment Variable)",
                    value="",
                    visible=False,
                    interactive=True,
                    type="password",
                )
                sbert_model_textbox = gr.Textbox(
                    label="Sentence Transformers Model Name",
                    value="all-MiniLM-L6-v2",
                    interactive=True,
                    visible=True,
                )
                set_emb_btn = gr.Button(value="Set Embedding Model", visible=True)
                set_emb_btn.click(
                    fn=m.set_embedding_model,
                    inputs=[
                        embedding_model_dropdown,
                        sbert_model_textbox,
                        openai_api_key,
                        state,
                    ],
                    outputs=set_emb_btn,
                )
            gr.Markdown(
                value=" 📖 Click the Embed Documents button. \n Alternatively, you can upload a numpy array file (.npy or .npz) with precomputed document embeddings. The file should be generated using numpy.save() with the shape (n_docs, embedding_dim). It must be embedded using the same embedding model as selected above, because the queries will be embedded using the same model. "
            )
            with gr.Row(variant="panel"):
                upload_emb_choice = gr.Radio(
                    choices=["Embed Documents", "Upload Precomputed Embeddings"],
                    label="Embedding Options",
                    visible=True,
                    interactive=True,
                )

                embed_btn = gr.Button("Embed Documents", visible=False)
                input_embedding_uploader = gr.File(
                    file_count="single",
                    file_types=[".npz", ".npy"],
                    label="Precomputed Embeddings",
                    visible=False,
                )
                emb_result_txtbox = gr.Textbox(
                    value="",
                    label="Embedding Progress",
                )
                emb_results_file = gr.File(visible=False)

            upload_emb_choice.change(
                fn=m.set_embedding_option,
                inputs=upload_emb_choice,
                outputs=[input_embedding_uploader, embed_btn],
                api_name=False,
            )

            input_file.change(
                fn=m.read_csv_cols,
                inputs=[input_file, state],
                outputs=[doc_col_selector, doc_id_col_selector],
                api_name=False,
            )

            input_embedding_uploader.change(
                fn=m.read_input_embedding,
                inputs=[input_embedding_uploader, state],
                outputs=[emb_result_txtbox, embed_btn],
                api_name=False,
            )
            embedding_model_dropdown.change(
                fn=m.toggle_embedding_model_visibility,
                inputs=[
                    embedding_model_dropdown,
                ],
                outputs=[sbert_model_textbox, openai_api_key, set_emb_btn],
                api_name=False,
            )
            sbert_model_textbox.change(
                fn=m.reset_set_emb_btn,
                outputs=[set_emb_btn],
            )
            # embedding
            openai_api_key.change(
                fn=m.set_openai_api_key,
                inputs=[openai_api_key, state],
                outputs=[emb_result_txtbox],
                api_name=False,
            )
            embed_btn.click(
                fn=m.embed_df,
                inputs=state,
                outputs=[emb_result_txtbox, emb_results_file, emb_results_file],
                api_name=False,
            )

        # Tab 2 =========================================================================================================
        with gr.Tab("2. Define Dimensions and Semantic Search"):
            gr.Markdown(
                value=" 📖 Move the sliders to set the number of dimensions and the number of results in each round of semantic search. Make sure that there is no empty dimension. ",
                label="",
            )
            n_dim_slider = gr.Slider(
                1,
                10,
                step=1,
                value=4,
                interactive=True,
                label="Number of dimensions",
            )
            n_results_slider = gr.Slider(
                10,
                200,
                step=5,
                value=10,
                interactive=True,
                label="Number of results in search",
            )

            gr.Markdown(
                value=" 📖 Enter the names of the dimensions and seed search queries. Then click 'Search Dimension' to search for relevant documents in the corpus. \nCopy & paste the relevant documents to query box (and/or edit) to conduct next round of search. You can use multiple queries in each dimension, with each query entered on its own line. ",
                label="",
            )

            for i in range(10):
                with gr.Row(variant="panel") as a_row:
                    all_rows_dims.append(a_row)
                    if i < 4:
                        a_row.visible = True
                    else:
                        a_row.visible = False
                    # 3 columns: query box, search button, search results
                    with gr.Column(scale=4, min_width=400):
                        if i < 4:
                            # default 4 dimensions visible
                            all_dim_name_boxes.append(
                                gr.Textbox(
                                    lines=1,
                                    max_lines=1,
                                    interactive=True,
                                    placeholder="e.g. " + CVFDemo().dim_names[i],
                                    value="",
                                    label=f"Dimension {i + 1} Name (Optional)",
                                    visible=True,
                                ),
                            )
                            all_search_query_boxes.append(
                                gr.Textbox(
                                    lines=5,
                                    interactive=True,
                                    label=f"Query (Seed) Sentences for Dimension {i + 1}. One per line. (Required)",
                                    value="",
                                    placeholder="e.g. " + CVFDemo().dim_seeds[i],
                                    visible=True,
                                )
                            )
                        else:
                            all_dim_name_boxes.append(
                                gr.Textbox(
                                    lines=1,
                                    max_lines=1,
                                    placeholder=None,
                                    interactive=True,
                                    label=f"Dimension {i + 1} Name (Optional)",
                                    value="",
                                    visible=True,
                                ),
                            )
                            all_search_query_boxes.append(
                                gr.Textbox(
                                    lines=5,
                                    interactive=True,
                                    label=f"Query (Seed) Sentences for Dimension {i + 1}. One per line. (Required)",
                                    value="",
                                    placeholder=f"Dimension {i + 1} seed sentences, one per line",
                                    visible=True,
                                )
                            )

                    with gr.Column(scale=1):
                        # search button and result textbox
                        all_search_btns.append(
                            gr.Button(
                                f"Search Dimension {i + 1}",
                                visible=True,
                            )
                        )

                    with gr.Column(scale=4, min_width=400):
                        all_search_results.append(
                            gr.Textbox(
                                value="",
                                label=f"Search Results for Dimension {i + 1}. Copy and paste relevant sentences into the query box to the left.",
                                visible=True,
                            )
                        )

            n_dim_slider.change(
                fn=m.toggle_row_vis,
                inputs=[n_dim_slider, state],
                outputs=all_rows_dims,
                api_name=False,
            )
            for box in all_search_query_boxes:
                box.change(
                    fn=m.toggle_row_vis,
                    inputs=[n_dim_slider, state],
                    api_name=False,
                )

            for dim_i, btn in enumerate(all_search_btns):
                btn.click(
                    fn=m.semantic_search,
                    inputs=[all_search_query_boxes[dim_i], n_results_slider, state],
                    outputs=all_search_results[dim_i],
                    api_name=False,
                )
            gr.Markdown(
                value=" 📖 After defining the dimensions with final context-specific queries, click 'Embed Queries and Save Dimensions' button below to embed the queries and save the definitions. Each dimension must contain at least one final query.",
                label="",
            )
            save_dim_button = gr.Button("Embed Queries and Save Dimensions")
            dim_define_results = gr.Textbox(visible=False, label="")
            dimension_def_file_download = gr.File(visible=False)

        # Tab 3 =========================================================================================================
        with gr.Tab("3. Define Scales"):
            tab2_warn = gr.Markdown(
                value="<span style='color:red'>⚠️ Embed Queries and Save Dimensions button in Tab 2 must be clicked before defining scales.</span>",
                visible=True,
            )
            gr.Markdown(
                value=" 📖 Move the sliders to set the number of scales.",
                label="",
            )
            n_scale_slider = gr.Slider(
                1,
                10,
                step=1,
                value=2,
                interactive=True,
                label="Number of scales",
            )

            gr.Markdown(
                value=" 📖 Enter the names of the scales and select the relevant dimensions. Then click the Save Scales button to compute the scale embedding vectors. \n📖 Scales are linear combinations of dimensions. e.g., Safety (Scale) = Safe (Positive Dimension) - Danger (Negative Dimension); Efficiency (Scale) = Productivity (Positive Dimension) - Waste (Negative Dimension); Wellness (Scale) = Physical Health (Positive Dimension) + Mental Health (Positive Dimension) - Illness (Negative Dimension) - Stress (Negative Dimension). \n📖 Each scale is computed by first averaging the dimension embedding vectors in the positive and negative dimensions, and then taking the difference. Each scale must contain at least one positive dimension; the negative dimension is optional. ",
                label="",
            )
            for i in range(10):
                with gr.Row(variant="panel") as a_row:
                    all_rows_scale.append(a_row)
                    if i >= 2:
                        a_row.visible = False
                        with gr.Column(scale=2, min_width=400):
                            all_scale_name_boxes.append(
                                gr.Textbox(
                                    lines=1,
                                    max_lines=1,
                                    interactive=True,
                                    placeholder=None,
                                    value="",
                                    label=f"Scale {i + 1} Name",
                                    visible=True,
                                ),
                            )
                    else:
                        a_row.visible = True
                        with gr.Column(scale=2, min_width=400):
                            all_scale_name_boxes.append(
                                gr.Textbox(
                                    lines=1,
                                    max_lines=1,
                                    interactive=True,
                                    placeholder="e.g. "
                                    + list(CVFDemo().scales.keys())[i],
                                    value="",
                                    label=f"Scale {i + 1} Name",
                                    visible=True,
                                ),
                            )
                        # 3 columns: scale name, positive dimension, negative dimension
                    with gr.Column(scale=4):
                        all_scale_pos_selector.append(
                            gr.Dropdown(
                                interactive=True,
                                multiselect=True,
                                label=f"Positive Dimensions for Scale {i + 1} (Required)",
                                value=None,
                                visible=True,
                                allow_custom_value=True,
                            )
                        )
                    with gr.Column(scale=4):
                        all_scale_neg_selector.append(
                            gr.Dropdown(
                                interactive=True,
                                multiselect=True,
                                label=f"Negative Dimensions for Scale {i + 1} (Optional)",
                                value=None,
                                visible=True,
                                allow_custom_value=True,
                            )
                        )
            n_scale_slider.change(
                fn=m.toggle_row_vis_scales,
                inputs=[n_scale_slider, state],
                outputs=all_rows_scale,
                api_name=False,
            )
            for box in all_scale_name_boxes:
                box.change(
                    fn=m.toggle_row_vis_scales,
                    inputs=[n_scale_slider, state],
                    api_name=False,
                )
            for box in all_scale_pos_selector:
                box.change(
                    fn=m.toggle_row_vis_scales,
                    inputs=[n_scale_slider, state],
                    api_name=False,
                )
            for box in all_scale_neg_selector:
                box.change(
                    fn=m.toggle_row_vis_scales,
                    inputs=[n_scale_slider, state],
                    api_name=False,
                )
            save_scale_button = gr.Button("Save Scales")
            scale_define_results = gr.Textbox(visible=False, label="")
            scale_def_file_download = gr.File(visible=False)

            save_dim_button.click(
                fn=m.save_dims,
                inputs=[state] + all_dim_name_boxes + all_search_query_boxes,
                outputs=all_scale_pos_selector
                + all_scale_neg_selector
                + [dim_define_results]
                + [dimension_def_file_download] * 2
                + [tab2_warn],
            )

        # Tab 4 =========================================================================================================
        with gr.Tab("4. Measurement"):
            tab3_warn = gr.Markdown(
                value="<span style='color:red'>⚠️ Save Scales button in Tab 3 must be clicked before measurement.</span>",
                visible=True,
            )
            save_scale_button.click(
                fn=m.save_scales,
                inputs=[state]
                + all_scale_name_boxes
                + all_scale_pos_selector
                + all_scale_neg_selector,
                outputs=[scale_define_results]
                + [scale_def_file_download] * 2
                + [tab3_warn],
            )

            gr.Markdown(
                value=" 📖 Click the Measure Documents Using Semantic Projection button to score each document. The output file will contain the document ID and the scores for each scale.",
                label="",
            )
            with gr.Row(variant="panel"):
                single_subspace_radio_btn = gr.Radio(
                    choices=["Yes", "No"],
                    value="No",
                    label="Single subspace: Select 'Yes' if all k scales span a single k-d subspace (recommended for reducing correlation when scales are similar in meaning). Select 'No' to treat each scale as a separate subspace.",
                    visible=True,
                )
                whitening_radio_btn = gr.Radio(
                    choices=["Yes", "No"],
                    value="Yes",
                    label="Whitening Output: Select 'Yes' to decorrelate the scores after semantic projection (recommended if the scales are theoretically orthogonal). ",
                    visible=True,
                )
            measure_button = gr.Button("Measure Documents Using Semantic Projection")
            measure_result = gr.Textbox(visible=False, label="")

            measure_results_file = gr.File(visible=False)
            measure_button.click(
                fn=m.measure_docs,
                inputs=[
                    single_subspace_radio_btn,
                    whitening_radio_btn,
                    state,
                ],
                outputs=[measure_result, measure_results_file, measure_results_file],
            )
        example_btn.click(
            fn=m.load_example_dataset,
            inputs=state,
            outputs=[
                example_row,
                example_file_download,
                example_emb_download,
                upload_emb_choice,
                emb_result_txtbox,
                doc_col_selector,
                doc_id_col_selector,
                embedding_model_dropdown,
                sbert_model_textbox,
            ]
            + all_dim_name_boxes
            + all_search_query_boxes
            + all_scale_name_boxes
            + all_scale_pos_selector
            + all_scale_neg_selector
            + [n_dim_slider, n_scale_slider],
        )

    # launch gradio =========================================================================================================
    demo.queue()
    if username is None or password is None:
        auth = None
    else:
        auth = (username, password)
    if mode == "public":
        demo.launch(
            share=True,
            auth=auth,
            server_name="0.0.0.0",
            favicon_path=path_mgt.sample_data_dir / "favicon.png",
            **kwargs,
        )
    elif mode == "local":
        demo.launch(
            favicon_path=path_mgt.sample_data_dir / "favicon.png",
            **kwargs,
        )
    else:
        print("Invalid mode. Please choose from 'public' or 'local'.")


if __name__ == "__main__":
    fire.Fire(run_gui)
