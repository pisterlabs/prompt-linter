from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import re

import numpy as np

import warnings
warnings.filterwarnings('ignore')
 


class IBLBoosting():
    def __init__(
        self, 
        llm_model_name, 
        params
        ):
        self.llm_model_name = llm_model_name
        self.llm_model = OpenAI(temperature=0, model_name = self.llm_model_name)

        #self.llm_model = llm_model,
        self.columns_name = params['columns_name']
        self.model_code = None


    def fit(self, x, y, model_name, file_path=None):
        print("> Start of model creating.")
        df = x.copy()

        df['target'] = y

        # Determine whether binary or multivalued classification is used
        if len(df['target'].unique()) == 2:
            task_type = 'binary classification'
            output_code = 'y = 1 / (1 + np.exp(-y))'
        else:
            task_type = 'multi-class classification'

        # Obtaining data types
        data_type = ', '.join(df.dtypes.astype(str))



        # Create a string dataset
        dataset = []
        for index, row in df.iterrows():
            row_as_str = [str(item) for item in row.tolist()] 
            dataset.append(','.join(row_as_str))
        dataset_str = '\n'.join(dataset)


        # column name
        if self.columns_name:
            col_name = ', '.join(df.columns.astype(str))
            col_option = ''

        else:
            # serial number
            df.columns = range(df.shape[1])
            col_name = ', '.join(df.columns.astype(str))
            col_option = 'df.columns = range(df.shape[1])'



        create_prompt = """
        Please create your code in compliance with all of the following conditions. Output should be code only. Do not enclose the output in ``python ``` or the like.
        ・Analyze the large amount of data below and create a {task_type_} code to accurately predict "target".
        ------------------
        {dataset_str_}
        ------------------
        ・Each data type is as follows. If necessary, you can change the data type.
        ・Create code that can make predictions about new data based on logic from large amounts of input data without using machine learning models.
        ・If input is available, the column names below should also be used to help make decisions when creating the predictive model. Column Name:{col_name_}
        ・Create a code like the following. Do not change the input or output format.
        ・If {col_option_} is not blank, add it after 'df = x.copy()'.
        ・You do not need to provide examples.
        ------------------
        import numpy as np

        def predict(x):
            df = x.copy()

            output = []
            for index, row in df.iterrows():


                # Feature creation and data preprocessing


                {output_code_}
                output.append(y)

            output = np.array(output)
                
            return output
        """.format(
            task_type_ = task_type,
            dataset_str_ = dataset_str,
            model_name_ = model_name,
            col_name_ = col_name,
            col_option_ = col_option,
            output_code_ = output_code
            )

        #print(create_prompt)

        with get_openai_callback() as cb:
            model_code = self.llm_model(create_prompt)
            print(cb)


        # Save to File
        if file_path != None:
            with open(file_path + f'{model_name}.py', mode='w') as file:
                file.write(model_code)


        self.model_code = model_code

        return model_code

    def predict(self, x):
        if self.model_code is None:
            raise Exception("You must train the model before predicting!")

        code = self.model_code

        # = re.search(r'def (\w+)', function_string).group(1)
        #code = self.model_code + '\n'# + f'model = model({x})'
        exec(code, globals())

        #model = namespace["code"]
        
        y = predict(x)

        return y




    def interpret(self):
        if self.model_code is None:
            raise Exception("You must train the model before interpreting!")

        interpret_prompt = """
        Refer to the code below and explain how you are going to process the data and make predictions.
        The only part to explain is the part where the data is processed.
        Do not explain df = x.copy().
        Please output the data in bulleted form.
        Please tell us what you can say based on the whole process.
        ------------------
        {model_code_}
        """.format(
            model_code_ = self.model_code
        )

        with get_openai_callback() as cb:
            output = self.llm_model(interpret_prompt)
            print(cb)


        return output