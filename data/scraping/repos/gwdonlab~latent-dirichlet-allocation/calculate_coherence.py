import os, json
import argparse as ap
import numpy as np
from ogm.trainer import TextTrainer
from ogm.utils import text_data_preprocess
from gensim.models import CoherenceModel


def get_args():
    p = ap.ArgumentParser()
    p.add_argument("filepath", help="Path to experiment's JSON file")
    p.add_argument(
        "--measure",
        help="Which coherence measure should be used? Defaults to 'all'",
        default="all",
        choices={"u_mass", "c_uci", "c_npmi", "all"},
    )
    p.add_argument(
        "--save_models",
        help="If set, this will actually save the coherence model rather than just writing the calculated score",
        action="store_true",
    )
    return p.parse_args()


def main(args):
    with open(args.filepath, "r") as infile:
        setup_dict = json.load(infile)

    # Look for input file at path and DATA_DIR if it's not there
    if not os.path.isfile(setup_dict["data_path"]):
        setup_dict["data_path"] = os.getenv("DATA_DIR") + "/" + setup_dict["data_path"]

    # Read in data and run the ogm preprocessing on it
    preprocessed_data = text_data_preprocess(setup_dict, output=False)
    trainer = TextTrainer()
    trainer.data = preprocessed_data
    topic_quants = range(setup_dict["min_topics"], setup_dict["max_topics"] + 1)
    text_key = setup_dict["text_key"]
    n_trials = setup_dict["n_trials"]
    experiment_name = setup_dict["name"]

    # Loop through different topic quantities
    for num_topics in topic_quants:

        # Load previous metadata
        with open(
            os.getenv("MODEL_DIR")
            + "/"
            + experiment_name
            + "/"
            + str(num_topics)
            + "topics/metadata.json",
            "r",
        ) as infile:
            metadata = json.load(infile)

        # For each topic quantity, run n_trials experiments
        coherences = {}
        for i in range(n_trials):
            model_savepath = (
                os.getenv("MODEL_DIR")
                + "/"
                + experiment_name
                + "/"
                + str(num_topics)
                + "topics/model_"
                + str(i)
            )
            trainer.load_model("lda", model_savepath + "/lda.model")

            # Make a coherence model for this LDA model
            if args.measure == "all":
                to_measure = {"u_mass", "c_uci", "c_npmi"}
            else:
                to_measure = {args.measure}

            for m in to_measure:
                cm = CoherenceModel(
                    model=trainer.model,
                    corpus=trainer.corpus,
                    texts=trainer.get_attribute_list(text_key),
                    coherence=m,
                )
                coherence = cm.get_coherence()

                if m not in coherences:
                    coherences[m] = []

                if np.isfinite(coherence):
                    coherences[m].append(coherence)
                else:
                    coherences[m].append(None)

                metadata["model_" + str(i)]["coherence_" + m] = coherences[m]

                if args.save_models:
                    cm.save(model_savepath + "/coherence_" + m + ".model")
                print("Finished", m, "coherence for trial", i, "in n_topics", num_topics)

        # Save information about the coherence scores overall
        for m in to_measure:
            c = np.array([x for x in coherences[m] if x is not None])
            metadata["aggregated"] |= {
                "avg_coherence_" + m: np.mean(c),
                "coherence_stdev_" + m: np.std(c),
                "coherence_variance_" + m: np.var(c),
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
    main(get_args())
