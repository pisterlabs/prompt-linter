from langchain.callbacks import get_openai_callback
import numpy as np
import pandas as pd
from importlib import resources

from ..utils import preprocessing

import warnings
warnings.filterwarnings('ignore')


class IBLModel():
    def __init__(
        self, 
        llm_model,
        params
        ):
        self.llm_model = llm_model
        self.columns_name = params['columns_name']
        self.objective = params['objective']
        self.code_model = None


    def fit(self, x, y, prompt = None, model_name = None, file_path = None):
        df = x.copy()
        df['target'] = y

        # Obtaining data types
        data_type = ', '.join(df.dtypes.astype(str))

        # Create a string dataset
        dataset_str = preprocessing.text_converter(df)

        # column name
        col_name, col_option = preprocessing.columns_name(self.columns_name, df)
     
        # create prompt
        if prompt == None:
            if self.objective == 'regression':
                with resources.open_text('iblm.iblmodel.prompt', 'regression.txt') as file:
                    prompt = file.read()
            elif self.objective == 'classification':
                with resources.open_text('iblm.iblmodel.prompt', 'classification_2.txt') as file:
                    prompt = file.read()

        create_prompt = prompt.format(
            dataset_str_ = dataset_str,
            data_type_ = data_type,
            col_name_ = col_name,
            col_option_ = col_option
            )

        code_model = self.llm_model(create_prompt)

        # prompt modification
        modification_prompt = preprocessing.prompt_modification(code_model)
        code_model = self.llm_model(modification_prompt)
        self.code_model = code_model

        # Save to File
        preprocessing.save_codemodel(file_path, model_name, self.code_model)
    
        return self.code_model



    def predict(self, x):
        if self.code_model is None:
            raise Exception("You must train the model before predicting!")

        code = self.code_model
        exec(code, globals())
        y = predict(x)
        return y



    def interpret(self):
        if self.code_model is None:
            raise Exception("You must train the model before interpreting!")

        interpret_prompt = preprocessing.interpret_codemodel(self.code_model)
        output = self.llm_model(interpret_prompt)
        return output