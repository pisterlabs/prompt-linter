import pickle
import pandas as pd
import numpy as np
import lightgbm as lgbm
from const import OUTPUT_DIR, INPUT_DIR
from langchain.tools import BaseTool
from models.make_data import make_forecast_train_data, make_forecast_test_data
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from utils import text_processing


class LgbmtrainTool(BaseTool):
    name = "lgbm_train_tool"
    description = """It is useful for learning LightGBM. The input should be the column name of the target variable."""

    def _run(self, query: str) -> str:
        """Use the tool."""
        print(query)
        query = text_processing(query)
        lgb_train, lgb_eval, num_class = make_forecast_train_data(query)
        # number of classes of the objective variable
        if num_class == 2:
            print("Binary Classification")
            params = {"task": "train", "boosting_type": "gbdt", "objective": "binary", "metric": "auc"}
        elif num_class <= 50:
            print("Multi Classification")
            params = {
                "task": "train",
                "boosting_type": "gbdt",
                "objective": "multiclass",
                "metric": "multi_logloss",
                "num_class": num_class,
            }
        else:
            print("Regression")
            params = {"task": "train", "boosting_type": "gbdt", "objective": "regression", "metric": "rmse"}

        lgbm_model = lgbm.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            callbacks=[
                lgbm.early_stopping(stopping_rounds=10, verbose=True),  # early_stopping用コールバック関数
                lgbm.log_evaluation(1),
            ],  # コマンドライン出力用コールバック関数
        )

        file = f"{OUTPUT_DIR}/trained_model.pkl"
        pickle.dump(lgbm_model, open(file, "wb"))

        split_df = pd.DataFrame(
            lgbm_model.feature_importance(importance_type="split"),
            index=lgb_train.data.columns,
            columns=["split_importance"],
        ).sort_values("split_importance", ascending=False)
        gain_df = pd.DataFrame(
            lgbm_model.feature_importance(importance_type="gain"),
            index=lgb_train.data.columns,
            columns=["gain_importance"],
        )
        importance = split_df.join(gain_df)
        importance.to_csv(f"{OUTPUT_DIR}/importance.csv")

        result = "LightGBM learning is complete."
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


class LgbminferenceTool(BaseTool):
    name = "lgbm_inference_tool"
    description = (
        """It is useful for inference using LightGBM. The input should be the column name of the objective variable."""
    )

    def _run(self, query: str) -> str:
        print(query)
        query = text_processing(query)

        file = f"{OUTPUT_DIR}/trained_model.pkl"
        lgbm_model = pickle.load(open(file, "rb"))

        df = pd.read_csv(f"{INPUT_DIR}/test.csv")

        categorical_features = []
        for i in df.columns:
            if df[i].dtypes == "O":
                df[i] = df[i].astype("category")
                categorical_features.append(i)

        # categorical features
        df[categorical_features] = df[categorical_features].astype("category")

        X = df.drop([query], axis=1)
        y = df[query]

        num_class = len(y.unique())
        # binary classification
        if num_class == 2:
            y_pred = lgbm_model.predict(X, num_interation=lgbm_model.best_iteration)
            roc_auc = roc_auc_score(y, y_pred)
            y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]
            pd.DataFrame(y_pred_binary).to_csv(f"{OUTPUT_DIR}/inference.csv")
            accuracy = accuracy_score(y, y_pred_binary)
            df_report = pd.DataFrame([roc_auc, accuracy], index=["roc_auc", "accuracy"], columns=["score"])
            df_report.to_csv(f"{OUTPUT_DIR}/report.csv")

        # multiclass classification
        elif num_class <= 50:
            y_pred = lgbm_model.predict(X, num_interation=lgbm_model.best_iteration)
            y_pred_class = [np.argmax(pred) for pred in y_pred]
            pd.DataFrame(y_pred_class).to_csv(f"{OUTPUT_DIR}/inference.csv")
            accuracy = accuracy_score(y, y_pred_class)
            # conf_mat = confusion_matrix(y, y_pred_class)
            df_report = pd.DataFrame([accuracy], index=["accuracy"], columns=["score"])
            df_report.to_csv(f"{OUTPUT_DIR}/report.csv")

        # regression
        else:
            y_pred = lgbm_model.predict(X, num_interation=lgbm_model.best_iteration)
            pd.DataFrame(y_pred).to_csv(f"{OUTPUT_DIR}/inference.csv")
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            df_report = pd.DataFrame([mse, r2], index=["mse", "r2"], columns=["score"])
            df_report.to_csv(f"{OUTPUT_DIR}/report.csv")

        result = "LightGBM inference is complete."
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")
