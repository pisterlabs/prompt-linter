from imodelsx import LinearFinetuneClassifier, LinearNgramClassifier, AugGAMClassifier
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import openai
import pandas as pd
import argparse
from copy import deepcopy
import logging
import random
from collections import defaultdict
from os.path import join
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import imodels
import inspect
import os.path
from imodelsx import cache_save_utils
from skllm.config import SKLLMConfig
from skllm import MultiLabelZeroShotGPTClassifier


path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

openai.api_key = open("/home/chansingh/.OPENAI_KEY").read().strip()
SKLLMConfig.set_openai_key(openai.api_key)


def add_eval(r, y_test, y_pred, y_pred_proba):
    cls_report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    for k1 in ["macro"]:
        for k in ["precision", "recall", "f1-score"]:
            r[f"{k1}_{k}"].append(cls_report[k1 + " avg"][k])
    r["accuracy"].append(accuracy_score(y_test, y_pred))
    r["roc_auc"].append(roc_auc_score(y_test, y_pred_proba))
    return r


def get_classification_data(lab="categorization___chief_complaint", input_text='description'):
    # read data
    df = pd.read_pickle(join(path_to_repo, 'data/data_clean.pkl'))

    # prepare output
    classes = df[lab].explode()
    vc = classes.value_counts()

    # restrict to top classes
    top_classes = vc.index[vc.values >= 20]
    df[lab] = df[lab].apply(lambda l: [x for x in l if x in top_classes])

    # label binarizer
    # top classes put most frequent first and last (helps with zero-shot)
    top_classes = top_classes[::2].tolist(
    ) + top_classes[1::2].tolist()[::-1]
    le = MultiLabelBinarizer(classes=top_classes)
    y = le.fit_transform(df[lab])

    # input text
    # set up text for prediction
    if input_text == 'raw_text':
        X = df["paper___raw_text"]
    elif input_text == 'description':
        def get_text_representation(row):
            # return f"""- Title: {row["title"]}
            # - Description: {row["description"]}
            # - Predictor variables: {str(row["feature_names"])[1:-1]}"""
            return f"""{row["title"]}. {row["description"]}."""  # Keywords: {str(row["info___keywords"])[1:-1]}"""
        X = df.apply(get_text_representation, axis=1)

    idxs = X.notna()
    X = X[idxs].tolist()
    y = y[idxs]

    # train test split
    return X, y, le.classes_, le
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)
    # return X_train, X_test, y_train, y_test, le.classes_


def get_model(model_name="decision_tree", random_state=42, class_name=None, input_text='raw_text'):
    if model_name == "decision_tree":
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("clf", DecisionTreeClassifier(random_state=random_state)),
            ]
        )
    elif model_name == "random_forest":
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("clf", RandomForestClassifier(random_state=random_state)),
            ]
        )
    # elif model_name == 'figs':
    #     return Pipeline(
    #         [

    #         ]
    #     )
    elif model_name == "logistic":
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                (
                    "clf",
                    # MultiOutputClassifier(
                    LogisticRegressionCV(random_state=random_state)
                    # ),
                ),
            ]
        )
    elif model_name == "aug-linear":
        return AugGAMClassifier(
            checkpoint="bert-base-uncased",
            normalize_embs=False,
            random_state=random_state,
            cache_embs_dir=os.path.expanduser(
                join(os.path.expanduser("~/.cache_mdcalc_embeddings"),
                     class_name, input_text)
            ),
            ngrams=2,
        )
    elif model_name == "bert-base-uncased":
        # pipe = MultiOutputClassifier(
        return LinearFinetuneClassifier(
            checkpoint="bert-base-uncased",
            normalize_embs=True,
            random_state=random_state,
            cache_embs_dir=join(os.path.expanduser(
                "~/.cache_mdcalc_embeddings"), class_name, input_text),
        )
    elif model_name == 'zero-shot':
        return MultiLabelZeroShotGPTClassifier(
            max_labels=5, openai_model="gpt-4-0314")


# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument(
        "--label_name",
        type=str,
        default="categorization___chief_complaint",
        choices=["categorization___chief_complaint",
                 "categorization___specialty",
                 "categorization___purpose",
                 "categorization___system",
                 "categorization___disease",],
        help="name of label",
    )
    parser.add_argument(
        '--input_text',
        type=str,
        default='raw_text',
        help='input text to use'
    )

    # training misc args
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(path_to_repo, "results"),
        help="directory for saving",
    )

    # model args
    parser.add_argument(
        "--model_name",
        type=str,
        default="decision_tree",
        help="name of model",
    )
    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)"""
    parser.add_argument(
        "--use_cache",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to check for cache",
    )
    return parser


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(
        deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir
    )

    if args.use_cache and already_cached:
        logging.info(f"cached version exists! Successfully skipping :)\n\n\n")
        exit(0)
    for k in sorted(vars(args)):
        logger.info("\t" + k + " " + str(vars(args)[k]))
    logging.info(f"\n\n\tsaving to " + save_dir_unique + "\n")

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    # torch.manual_seed(args.seed)

    # get data
    X, y, classes, le = get_classification_data(
        lab=args.label_name, input_text=args.input_text)

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique
    os.makedirs(save_dir_unique, exist_ok=True)
    # cache_save_utils.save_json(
    # args=args, save_dir=save_dir_unique, fname="params.json", r=r
    # )

    # fit + eval
    if not args.model_name == 'zero-shot':
        for i, c in enumerate(tqdm(classes)):
            m = get_model(
                args.model_name,
                random_state=42,
                class_name=c,
                input_text=args.input_text,
            )
            y_i = y[:, i]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_i, test_size=0.25, random_state=42, stratify=y_i
            )

            m.fit(X_train, y_train)
            # df['y_pred_train'].append(m.predict(X_train))
            y_pred = m.predict(X_test)
            y_pred_proba = m.predict_proba(X_test)[:, 1]
            # df['y_pred_test'].append(y_test)
            r = add_eval(r, y_test, y_pred, y_pred_proba)

    elif args.model_name == 'zero-shot':
        m = get_model(
            args.model_name,
            random_state=42,
            input_text=args.input_text,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        m.fit(None, [classes.tolist()])
        # df['y_pred_train'].append(m.predict(X_train))
        y_pred_strs = m.predict(X_test)
        y_pred = le.transform(y_pred_strs)
        y_pred_proba = y_pred
        for i in range(len(classes)):
            r = add_eval(r, y_test[:, i], y_pred[:, i], y_pred_proba[:, i])

    for k in ['macro_precision', 'macro_recall', 'macro_f1-score', 'accuracy', 'roc_auc']:
        r[f"mean_{k}"] = np.mean(r[k])

    # save results
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    # joblib.dump(model, join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")
