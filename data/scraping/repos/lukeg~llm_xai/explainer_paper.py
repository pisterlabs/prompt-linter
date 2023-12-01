import pickle

import torch
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate,)
import numpy as np
import random
np.int = int #fixing shap/numpy compatibility issue
from sklearn.metrics import classification_report
import shap
from matplotlib import pyplot as plt
from langchain.chains import LLMChain
from lime_stability.stability import LimeTabularExplainerOvr
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib
import langchain
from langchain.output_parsers.enum import EnumOutputParser
from enum import Enum
from progressbar import ProgressBar, Percentage, Bar, Timer, ETA, Counter
#cf. https://stackoverflow.com/a/53304527/5899161
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import fastshap
from torch import nn
import dice_ml
from anchor import anchor_tabular
from langchain.llms import VLLM
from SALib.sample import morris as morris_sample
from SALib.test_functions import Ishigami
from SALib.analyze import morris as morris_analyze
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
import tqdm
#langchain.verbose=True
def vicuna15(temperature=.7):
    model = "vicuna"
    llm = ChatOpenAI(model_name=model, openai_api_key="EMPTY", openai_api_base="http://localhost:8000/v1", max_tokens=150, verbose=True, temperature=temperature)
    return llm

def llama2(temperature=.4):
    model = "llama2"
    llm = ChatOpenAI(model_name=model, openai_api_key="EMPTY", openai_api_base="http://localhost:8000/v1", max_tokens=150,
                     temperature=temperature)
    return llm

def llama2_hf_70b(temperature = .4):
    #cf. https://www.pinecone.io/learn/llama-2/
    import torch
    import transformers
    from langchain.llms import HuggingFacePipeline

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_id = "/lu/tetyda/home/lgorski/llama/llama-2-70b-chat-hf/models--meta-llama--Llama-2-70b-chat-hf/snapshots/36d9a7388cc80e5f4b3e9701ca2f250d21a96c30/"
    model_config = transformers.AutoConfig.from_pretrained(model_id)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        config=model_config,
        quantization_config=bnb_config,
        device_map = "auto")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer, task="text-generation",
        temperature=temperature,
        max_new_tokens=150,
        repetition_penalty=1.1
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    return llm

def gpt4_azure(temperature=.3):
    import json
    import os
    with open("gpt4.json", encoding="utf-8") as credentials_file:
        credentials = json.load(credentials_file)

    llm = AzureChatOpenAI(
        openai_api_base=credentials["OPENAI_API_BASE"],
        openai_api_version=credentials["OPENAI_API_VERSION"],
        deployment_name="test-gpt4-32k",
        openai_api_key=credentials["OPENAI_API_KEY"],
        openai_api_type=credentials["OPENAI_API_TYPE"],
        max_tokens=150,
        temperature=temperature,
    )
    return llm

def grouper(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


#returns a list of violated rules
def predict_rules_only(X, features_closure, encoder : defaultdict):
    def analyze_rules(X):
        features = features_closure.tolist()
        violated = []
        #r1 (gender = f and age >= 60) or (gender = male and age >= 65)
        gender_idx = features.index("gender")
        age_idx = features.index("age")
        gender = encoder[gender_idx].inverse_transform([X[gender_idx]])[0]
        age = X[age_idx]
        if not((gender == "f" and X[age_idx] >= 60) or (gender == "m" and age >= 65)):
            violated.append(1)
        #r2 r2: at least four of the following features are "yes": paid_contribution_1, paid_contribution_2, paid_contribution_3, paid_contribution_4, paid_contribution_5
        paid_contribution_1_idx = features.index("paid_contribution_1")
        paid_contribution_2_idx = features.index("paid_contribution_2")
        paid_contribution_3_idx = features.index("paid_contribution_3")
        paid_contribution_4_idx = features.index("paid_contribution_4")
        paid_contribution_5_idx = features.index("paid_contribution_5")
        paid_contribution_1 = encoder[paid_contribution_1_idx].inverse_transform([X[paid_contribution_1_idx]])[0]
        paid_contribution_2 = encoder[paid_contribution_2_idx].inverse_transform([X[paid_contribution_2_idx]])[0]
        paid_contribution_3 = encoder[paid_contribution_3_idx].inverse_transform([X[paid_contribution_3_idx]])[0]
        paid_contribution_4 = encoder[paid_contribution_4_idx].inverse_transform([X[paid_contribution_4_idx]])[0]
        paid_contribution_5 = encoder[paid_contribution_5_idx].inverse_transform([X[paid_contribution_5_idx]])[0]
        paid_contributions = sum([1 if elem == "yes" else 0 for elem in [paid_contribution_1, paid_contribution_2, paid_contribution_3, paid_contribution_4, paid_contribution_5]])
        if not (paid_contributions >= 4):
            violated.append(2)
        #r3 r3: is_spouse=yes
        is_spouse_idx = features.index("is_spouse")
        is_spouse = encoder[is_spouse_idx].inverse_transform([X[is_spouse_idx]])[0] == "True"
        if not (is_spouse == True):
            violated.append(3)
        #r4 is_absent=no
        is_absent_idx = features.index("is_absent")
        is_absent = encoder[is_absent_idx].inverse_transform([X[is_absent_idx]])[0] == "True"
        if not (is_absent == False):
            violated.append(4)
        #r5 it is not true that capital_resources >= 3000
        capital_resources_idx = features.index("capital_resources")
        capital_resources = X[capital_resources_idx]
        if capital_resources >= 3000:
            violated.append(5)
        # r6: (patient_type= in and distance_to_hospital < 50) or (patient_type=out and distance_to_hospital >= 50)
        patient_type_idx = features.index("patient_type")
        distance_to_hospital_idx = features.index("distance_to_hospital")
        patient_type = encoder[patient_type_idx].inverse_transform([X[patient_type_idx]])[0]
        distance_to_hospital = X[distance_to_hospital_idx]
        if not ((patient_type == "in" and distance_to_hospital < 50) or (patient_type == "out" and distance_to_hospital >= 50)):
            violated.append(6)
        return violated

    def inner(X, violated_rules=None):
        if violated_rules==None:
            violated_rules=[]
        result = []
        for row in X:
            violated = analyze_rules(row)
            violated_rules.append(violated)
            if len(violated) == 0:
                result.append(1)
            else:
                result.append(0)
        return np.array(result)

    return inner


def predict_rules_simplified_only(X, features_closure, encoder : defaultdict):
    def analyze_rules(X):
        features = features_closure.tolist()
        violated = []
        #r1 (gender = f and age >= 60) or (gender = male and age >= 65)
        gender_idx = features.index("gender")
        age_idx = features.index("age")
        gender = encoder[gender_idx].inverse_transform([X[gender_idx]])[0]
        age = X[age_idx]
        if not((gender == "f" and X[age_idx] >= 60) or (gender == "m" and age >= 65)):
            violated.append(1)

        # r2: (patient_type= in and distance_to_hospital < 50) or (patient_type=out and distance_to_hospital >= 50)
        patient_type_idx = features.index("patient_type")
        distance_to_hospital_idx = features.index("distance_to_hospital")
        patient_type = encoder[patient_type_idx].inverse_transform([X[patient_type_idx]])[0]
        distance_to_hospital = X[distance_to_hospital_idx]
        if not ((patient_type == "in" and distance_to_hospital < 50) or (patient_type == "out" and distance_to_hospital >= 50)):
            violated.append(2)
        return violated

    def inner(X, violated_rules=None):
        if violated_rules==None:
            violated_rules=[]
        result = []
        for row in X:
            violated = analyze_rules(row)
            violated_rules.append(violated)
            if len(violated) == 0:
                result.append(1)
            else:
                result.append(0)
        return np.array(result)

    return inner



import time
def predict_rules(chain, features, encoder : defaultdict, configuration=None, save_reply=False, memory={}):
    def predict_rules_inner(X : np.ndarray, output = None):
        results = []
        widgets = [' [', Percentage(), '] ', Bar(), ' (', Timer(), ') ', ETA(), ' ', Counter(), ' of ', str(len(X))]
        pbar = ProgressBar(widgets=widgets, maxval=len(X)).start()
        counter = 0
        X=np.array(X, dtype=object)

        for index, encoding in encoder.items():
            inversed = encoding.inverse_transform(X[:, index].astype(int))
            X[:, index] = inversed

        for row in X:
            if row.tobytes() in memory:
                classification = memory[row.tobytes()]
            else:
                text=",".join([str(elem) for elem in row])
                classification = chain(text)
                if output is not None:
                    output += [classification]
                if save_reply:
                    with open(f"log_{configuration.model_factory}.txt", "a") as log:
                        log.write(classification + "\n")
                classification=classification["text"]
                cleaned = classification.strip().replace(".", "").lower()
                if "granted" in cleaned:
                    results.append(1)
                elif "denied" in cleaned:
                    results.append(0)
                else: #answer not fitting the template
                    results.append(2)
            counter += 1
            pbar.update(counter)
            if configuration.throttle:
                time.sleep(configuration.throttle)
        pbar.finish()
        return np.array(results)
    return predict_rules_inner

# def lime_explainer(train, test, predict, feature_names, encoder):
#     categorical_features = list(sorted(encoder.keys()))
#     categorical_names = { key : list(encoder[key].classes_) for key in categorical_features}
#     explainer = LimeTabularExplainerOvr(train, feature_names=feature_names, categorical_features=categorical_features,
#                                         categorical_names=categorical_names,
#                                         class_names=["not granted", "granted", "unknown"])
#     print(explainer.explain_instance(np.array(test[0]), predict).as_list())



import random
def shap_explainer(train, test, y_train, y_test, predict, features, encoder, configuration):
    explainer = shap.KernelExplainer(model=predict, data=shap.sample(train, 100))

    #test=pd.read_csv(r"data/welfare_dataset/DiscoveringTheRationaleOfDecisions/datasets/confused_gpt4.csv").drop(columns=["eligible"])
    shap_values = explainer.shap_values(test)
    if configuration.saveout:
        np.save(f'shap_values_{configuration.model_factory}.npy', shap_values)
    print(shap_values)
    #shap.summary_plot(shap_values, show=False, feature_names=features, class_names=["not granted", "granted", "unknown"])
    #plt.savefig('vis.png')

def morris_explainer(train, test, y_train, y_test, predict, features, encoder, configuration):
    from interpret.blackbox import MorrisSensitivity
    msa = MorrisSensitivity(predict, test, feature_names=features, num_resamples=10, num_levels=2)
    print(msa)


def anchor_explainer(train, test, y_train, y_test, predict, features, encoder, configuration):
    explainer = anchor_tabular.AnchorTabularExplainer(
        ["not granted", "granted", "unknown"],
        features,
        train,
        { key: value.classes_ for key, value in encoder.items() })
    explanations = []
    for test_instance_idx in range(test.shape[0]):
        print (f"calculating {test_instance_idx} of {len(test)}")
        explanation = explainer.explain_instance(test[test_instance_idx], predict, threshold=0.95)
        print("Anchor: %s" % (' AND '.join(explanation.names())))
        print('Precision: %.2f' % explanation.precision())
        print('Coverage: %.2f' % explanation.coverage())
        explanations.append(explanation)

    if configuration.saveout:
        with open("anchor_explanations.pkl", "wb") as output:
            pickle.dump(explanations, output)

def counterfactual_explainer(train, test, y_train, y_test, predict, features, encoder, configuration):
    pd_train_x = pd.DataFrame(train, columns=features)
    pd_train_y = pd.DataFrame(y_train, columns=["eligible"])
    pd_test_x = pd.DataFrame(test, columns=features)
    pd_test_y = pd.DataFrame(y_test, columns=["eligible"])
    dice_data = pd.concat([pd_train_x, pd_train_y], axis=1, join='inner')
    dice_data = dice_data.infer_objects()
    continuous_features = [ features[idx] for idx in encoder.keys()]

    dice_dataset = dice_ml.Data(dataframe=dice_data, outcome_name='eligible', continuous_features=continuous_features)
    dice_model = dice_ml.Model(model=predict, backend="sklearn")
    exp = dice_ml.Dice(dice_dataset, dice_model, method="random")
    explanation = exp.generate_counterfactuals(pd.DataFrame(pd_test_x, columns=features), total_CFs=3, desired_class="opposite")
    print(explanation)
    #print(exp.visualize_as_dataframe(show_only_changes=True, display_sparse_df=False))

def define_command_line_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-factory', choices=["llama2_hf_70b",
        "llama2_transformers", "vicuna", "vicuna15", "vicuna_vllm", "llama2_vllm", "gpt4_azure", "llama2"
    ], default="vicuna")
    parser.add_argument('--dataset', type=str, default="A_2400.csv")
    parser.add_argument('--predict-function', choices = ["predict_rules", "predict_rules_only"], default="predict_rules")
    parser.add_argument('--system-template', type=str, default="system_template_6_conditions.txt")
    parser.add_argument('--xai', default=[], choices=["shap_explainer", "fastshap_explainer",
                                                      "fastshap2_explainer", "morris_explainer", "counterfactual_explainer"], action='append')
    parser.add_argument('--saveout', default=False, action='store_true')
    parser.add_argument('--classification-report', default=False, action='store_true')
    parser.add_argument('--fastshap-model-load', default=None, type=str)
    parser.add_argument('--fastshap-model-save', default=None, type=str)
    parser.add_argument('--drop-noise', default=False, action='store_true')
    parser.add_argument('--check-rules', default=False, action='store_true')
    parser.add_argument('--test-size', default=.2, type=float)
    parser.add_argument('--optimize-temperature', default=False, action='store_true')
    parser.add_argument('--throttle', default=0, type=int)
    parser.add_argument('--stability', default=False, action='store_true')
    parser.add_argument('--surrogate-model', default=None, type=str)
    parser.add_argument('--ablation-study', default=False, action='store_true')
    parser.add_argument('--confusion-study', default=False, action='store_true')
    return parser

def read_command_line_options(parser : argparse.ArgumentParser) -> argparse.Namespace:
    return parser.parse_args()



#this function does too much, split it later
def prepare_train_test_split(dataset, test_size=.2, drop_noise=False, random_state=42):
    df = pd.read_csv(dataset)
    encoding = defaultdict(LabelEncoder)
    mask = (df.dtypes == object) | (df.dtypes == bool)
    X = df.drop(columns=["eligible"], axis=1)
    if drop_noise:
        to_drop = [ col for col in X.columns if "noise" in col]
        X = X.drop(columns=to_drop, axis=1)
    X.loc[:, mask] = X.loc[:, mask].astype(str).apply(lambda s: encoding[X.columns.get_loc(s.name)].fit_transform(s))
    X_columns = X.columns
    #conversion to numpy array, because shap works with numpy arrays
    X = X.to_numpy()
    y = df["eligible"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X, y, X_train, y_train, X_test, y_test, X_columns, encoding




parser = define_command_line_options()
configuration = read_command_line_options(parser)

system_template = pathlib.Path(configuration.system_template).read_text()
system_prompt = SystemMessagePromptTemplate.from_template(system_template)

X, y, X_train, y_train, X_test, y_test, columns, encoding = prepare_train_test_split(configuration.dataset,
                                                                                     test_size=configuration.test_size,
                                                                                     drop_noise=configuration.drop_noise)
X_columns = ",".join([name for name in columns])
system_prompt = system_prompt.format(names=X_columns)
human_template = """{features}"""
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])



llm = globals()[configuration.model_factory]()
chain = LLMChain(llm=llm, prompt=chat_prompt)
predict = globals()[configuration.predict_function](chain, columns, encoding, configuration=configuration)


def create_llm_classifier(predict):
    def _create_llm_classifier(cls):
        from sklearn.base import BaseEstimator, ClassifierMixin
        class LLMEstimatorSK(BaseEstimator, ClassifierMixin, cls):
            def __init__(self, temperature=.7):
                self.temperature = temperature

            def fit(self, X, y):
                return self

            def predict(self, X):
                return predict(X)

        return LLMEstimatorSK

    return _create_llm_classifier


@create_llm_classifier(predict=predict)
class LLMEstimatorSK:
    def __init__(self, temperature=.7):
        self.temperature = temperature

    def fit(self, X, y):
        return self

    def predict(self, X):
        return predict(X)

def indices_tmp():
    indices = range(len(X_train))
    _, _, _, _, indices_train, indices_test = train_test_split(X_train, y_train, indices, test_size=configuration.test_size,
                                                                                     random_state=42, stratify=y_train)
    yield indices_train, indices_test

if configuration.optimize_temperature:
    from sklearn.model_selection import GridSearchCV

    grid = GridSearchCV(LLMEstimatorSK(), cv=indices_tmp(), param_grid={"temperature": [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]})
    grid.fit(X_train, y_train)
    print(grid.best_params_)
    scores = pd.DataFrame(grid.cv_results_)
    scores.to_excel(f"grid_search_{configuration.model_factory}_04_06.xlsx")

#rules consistency check
if configuration.check_rules:
    predict_rules=predict_rules_only(X_train, columns, encoding)
    for idx in range(len(X_train)):
        violated = analyze_rules(X_train[idx], columns, encoding)
        if (len(violated) == 0 and y_train[idx] == False) or (len(violated) > 0 and y_train[idx] == True):
            print(f"rule violation for: {X_train[idx]}")



if configuration.classification_report:
    y_pred = predict(X_test)
    print(classification_report(y_test, y_pred))

if configuration.stability:
    output = []
    for _ in range(5):
        predict(X_test, output=output)
    import json
    with open(f"stability_{configuration.model_factory}.json", "w") as output_file:
        json.dump(output, output_file)

if configuration.surrogate_model:
    from sklearn.tree import DecisionTreeClassifier

    surrogate_model = DecisionTreeClassifier(random_state=42)
    #get model answers
    answers = pd.read_json(configuration.surrogate_model)
    target = pd.DataFrame([1 if "granted" in text.lower() else 0 for text in answers["text"].tolist()])
    #get indices of the test set
    indices = range(len(X))
    _, _, _, _, indices_train, indices_test = train_test_split(X, y, indices, test_size=configuration.test_size,
                                                                                     random_state=42, stratify=y)
    print(len(target))
    surrogate_model.fit(X_train, target.iloc[indices_train])
    print(surrogate_model.feature_importances_)

    for xai in configuration.xai:
        globals()[xai](X_train, X_test, y_train, y_test, surrogate_model.predict, columns, encoding, configuration)

for xai in configuration.xai:
    globals()[xai](X_train, X_test, y_train, y_test, predict, columns, encoding, configuration)

if configuration.ablation_study:
    import json
    output=[]
    y_pred=predict(X_test, output=output)
    with open(f"ablation_{configuration.system_template}", "w") as classification_file, \
        open(f"ablation_{configuration.system_template}.json", "w") as json_file:
        classification_file.write(classification_report(y_test, y_pred))
        json.dump(output, json_file)

if configuration.confusion_study:
# ---tmp----

    class Status:
        def __init__(self):
            self.first_rule_violated = False,
            self.second_rule_violated = False,
            self.y_true = None,
            self.y_pred = None
            self.features = None
            self.answer = None

        def __str__(self):
            return f"first_rule_violated: {self.first_rule_violated}, second_rule_violated: {self.second_rule_violated}, y_true: {self.y_true}, y_pred: {self.y_pred}, features: {self.features}, gpt4: {self.answer}"

        def __repr__(self):
            return self.__str__()
        def __hash__(self):
            return hash((self.first_rule_violated, self.second_rule_violated, self.y_true,
                         self.y_pred))

        def __eq__(self, other):
            return (self.first_rule_violated, self.second_rule_violated, self.y_true,
                         self.y_pred) == (other.first_rule_violated, other.second_rule_violated, other.y_true,
                         other.y_pred)
    #
#additional modules
    statuses = set()
    y_=[1 if yy else 0 for yy in y_test]

    analyze_rules=predict_rules_simplified_only(X_test, columns, encoding)
    violations=[]
    analyze_rules(X_test, violations)
    x_test_set = []
    y_test_set = []
    for rule, violation, y_true in zip(X_test, violations, y_):
        if len(statuses) == 8:
            break
        status = Status()
        status.y_true = y_true
        output=[]
        status.y_pred = predict([rule], output=output)[0]
        status.features = rule
        status.first_rule_violated = 1 in violation
        status.second_rule_violated = 2 in violation
        status.answer = output[0]

        if status not in statuses:
            statuses.add(status)
            x_test_set += [rule]
            y_test_set += [y_]
            print(status)
            print("----- Found rule ----")
            with open("confusion_study.txt", "a") as file:
                file.write(str(status))
                file.write("\n")

    print("-----------------")
    print(statuses)
    print("---performing xai----")
    shap_explainer(X_train, np.array(x_test_set), y_train, y_test_set, predict, columns, encoding, configuration)
