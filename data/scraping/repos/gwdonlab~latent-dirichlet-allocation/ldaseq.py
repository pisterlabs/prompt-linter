import os, json
import argparse as ap
import numpy as np
from pandas import to_datetime
from ogm.trainer import TextTrainer
from ogm.utils import text_data_preprocess
from gensim.models import CoherenceModel


def get_setup_dict():
    p = ap.ArgumentParser()
    p.add_argument("filepath", help="Path to experiment's JSON file")
    a = p.parse_args()
    with open(a.filepath, "r") as infile:
        input_dict = json.load(infile)

    return input_dict


def main(setup_dict):

    # Look for input file at path and DATA_DIR if it's not there
    if not os.path.isfile(setup_dict["data_path"]):
        setup_dict["data_path"] = os.getenv("DATA_DIR") + "/" + setup_dict["data_path"]

    # Read in data and run the ogm preprocessing on it
    if "time_filter" not in setup_dict:
        raise ValueError("A time filter is required for training a sequential LDA")
    preprocessed_data = text_data_preprocess(setup_dict, output=False)

    # Add to Trainer object
    trainer = TextTrainer(log=setup_dict["name"] + str(setup_dict["min_topics"]) + ".log")
    trainer.data = preprocessed_data
    print("Found", trainer.data.shape[0], "posts")

    # Order chronologically and split by time window
    trainer.data["__ts"] = to_datetime(trainer.data[setup_dict["time_filter"]["time_key"]])
    ts_df = trainer.data.sort_values(by="__ts")[["__ts", setup_dict["time_filter"]["time_key"]]]
    ts_df = (
        ts_df.set_index("__ts")
        .resample(str(setup_dict["days_in_interval"]) + "D")
        .agg({setup_dict["time_filter"]["time_key"]: "count"})
        .reset_index()
    )
    docs_quants = [y for y in ts_df[setup_dict["time_filter"]["time_key"]]]
    time_labels = [x.strftime("%Y-%m-%d") for x in ts_df["__ts"]]

    # Load hyperparameters
    topic_quants = range(setup_dict["min_topics"], setup_dict["max_topics"] + 1)
    text_key = setup_dict["text_key"]
    experiment_name = setup_dict["name"]
    if "passes" in setup_dict:
        passes = setup_dict["passes"]
    else:
        passes = 10

    print("Training models for topic_nums:", topic_quants)

    # Loop through different topic quantities
    for num_topics in topic_quants:
        metadata = {}
        coherences = []

        # Create directory where model files will be saved
        model_savepath = (
            os.getenv("MODEL_DIR") + "/" + experiment_name + "/" + str(num_topics) + "topics"
        )
        os.makedirs(model_savepath, exist_ok=True)

        # Train model
        trainer.train_ldaseq(
            col=text_key,
            n_topics=num_topics,
            output_path=model_savepath + "/ldaseq.model",
            seq_counts=docs_quants,
            passes=passes,
        )

        # Loop through the different time slices and get coherence at each one
        for i, quantity in enumerate(docs_quants):
            model_output = trainer.model.dtm_coherence(i)
            cm = CoherenceModel(
                corpus=trainer.corpus,
                texts=trainer.get_attribute_list(text_key),
                topics=model_output,
                coherence="c_v",
                dictionary=trainer.dictionary,
            )

            # Save information about this time slice
            coherence = cm.get_coherence()
            coherences.append(coherence)
            metadata["time_" + str(i)] = {
                "coherence": coherence,
                "start_time": time_labels[i],
                "num_posts": quantity,
                "coherence_savepath": model_savepath + "/coherence_" + str(i) + ".model",
            }

            # Save coherence model
            cm.save(model_savepath + "/coherence_" + str(i) + ".model")

        # Save information about the coherence scores over all the time slices
        coherences = np.array(coherences)
        metadata["aggregated"] = {
            "avg_coherence": np.mean(coherences),
            "coherence_stdev": np.std(coherences),
            "coherence_variance": np.var(coherences),
            "topics": num_topics,
        }

        with open(
            os.getenv("MODEL_DIR")
            + "/"
            + experiment_name
            + "/"
            + str(num_topics)
            + "topics/metadata.json",
            "w",
        ) as output:
            json.dump(metadata, output)


if __name__ == "__main__":
    d = get_setup_dict()
    main(d)
