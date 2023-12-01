import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split

from langchain.tools import BaseTool


class LgbmtrainTool(BaseTool):
    name = "lgbm_train_tool"
    description = """useful to receive csv file name and learn LightGBM"""

    def _run(self, query: str) -> str:
      """Use the tool."""
      #global lgbm
      path = os.getcwd()
      df = pd.read_csv(f'{path}/{query}', index_col = 0)
      x = df.drop(['target'], axis = 1)
      y = df['target']

      x_train,x_valid,y_train,y_valid = train_test_split(x, y ,test_size = 0.2, random_state=3655)

      # categorical features
      categorical_features = []
      for i in df.columns:
          if df[i].dtypes == 'category':
              categorical_features.append(i)


      lgb_train = lgbm.Dataset(x_train,y_train,categorical_feature=categorical_features,free_raw_data=False)
      lgb_eval = lgbm.Dataset(x_valid,y_valid,reference=lgb_train,categorical_feature=categorical_features,free_raw_data=False)


      # number of classes of the objective variable
      num_class = len(df['target'].unique())
      if num_class == 2:
          params = {'task': 'train', 'boosting_type': 'gbdt','objective': 'binary', 'metric': 'auc', 'verbose': -1}
      elif num_class <= 50:
          params = {'task': 'train', 'boosting_type': 'gbdt','objective': 'multiclass', 'metric': 'multi_logloss','num_class': num_class, 'verbose': -1}
      else:
          params = {'task': 'train', 'boosting_type': 'gbdt','objective': 'regression','metric': 'rmse', 'verbose': -1}


      lgbm_model = lgbm.train(
          params,
          lgb_train,
          valid_sets=[lgb_train,lgb_eval],
          verbose_eval=10,
          #num_boost_round=1000,
          early_stopping_rounds= 10
          )
      

      file = f'{path}/trained_model.pkl'
      pickle.dump(lgbm_model, open(file, 'wb'))

      result = "LightGBMの学習が完了しました"
      return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


class LgbminferenceTool(BaseTool):
    name = "lgbm_inference_tool"
    description = """useful for receiving csv file name and making inferences in LightGBM"""

    def _run(self, query: str) -> str:
        """Use the tool."""
        path = os.getcwd()
        x = pd.read_csv(f'{path}/{query}', index_col = 0)

    

        file = f'{path}/trained_model.pkl'
        lgbm_model = pickle.load(open(file, 'rb'))

        y_pred = lgbm_model.predict(x, num_interation=lgbm_model.best_iteration)
        y_pred = pd.DataFrame(y_pred)
        y_pred.to_csv(f'{path}/inference.csv')


        result = "LightGBMの推論が完了しました" 
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


