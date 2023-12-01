import pandas as pd
import requests
# import tensorflow
import tensorflow as tf
#import keras 
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


"""Creates the Example and GPT classes for a user to interface with the OpenAI
API."""

import openai
import uuid


def set_openai_key(key):
    """Sets OpenAI key."""
    openai.api_key = key


class Example:
    """Stores an input, output pair and formats it to prime the model."""
    def __init__(self, inp, out):
        self.input = inp
        self.output = out
        self.id = uuid.uuid4().hex

    def get_input(self):
        """Returns the input of the example."""
        return self.input

    def get_output(self):
        """Returns the intended output of the example."""
        return self.output

    def get_id(self):
        """Returns the unique ID of the example."""
        return self.id

    def as_dict(self):
        return {
            "input": self.get_input(),
            "output": self.get_output(),
            "id": self.get_id(),
        }


class GPT:
    """The main class for a user to interface with the OpenAI API.

    A user can add examples and set parameters of the API request.
    """
    def __init__(self,
                 engine='davinci',
                 temperature=0.5,
                 max_tokens=100,
                 input_prefix="input: ",
                 input_suffix="\n",
                 output_prefix="output: ",
                 output_suffix="\n\n",
                 append_output_prefix_to_query=False):
        self.examples = {}
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.input_prefix = input_prefix
        self.input_suffix = input_suffix
        self.output_prefix = output_prefix
        self.output_suffix = output_suffix
        self.append_output_prefix_to_query = append_output_prefix_to_query
        self.stop = (output_suffix + input_prefix).strip()

    def add_example(self, ex):
        """Adds an example to the object.

        Example must be an instance of the Example class.
        """
        assert isinstance(ex, Example), "Please create an Example object."
        self.examples[ex.get_id()] = ex

    def delete_example(self, id):
        """Delete example with the specific id."""
        if id in self.examples:
            del self.examples[id]

    def get_example(self, id):
        """Get a single example."""
        return self.examples.get(id, None)

    def get_all_examples(self):
        """Returns all examples as a list of dicts."""
        return {k: v.as_dict() for k, v in self.examples.items()}

    def get_prime_text(self):
        """Formats all examples to prime the model."""
        return "".join(
            [self.format_example(ex) for ex in self.examples.values()])

    def get_engine(self):
        """Returns the engine specified for the API."""
        return self.engine

    def get_temperature(self):
        """Returns the temperature specified for the API."""
        return self.temperature

    def get_max_tokens(self):
        """Returns the max tokens specified for the API."""
        return self.max_tokens

    def craft_query(self, prompt):
        """Creates the query for the API request."""
        q = self.get_prime_text(
        ) + self.input_prefix + prompt + self.input_suffix
        if self.append_output_prefix_to_query:
            q = q + self.output_prefix

        return q

    def submit_request(self, prompt):
        """Calls the OpenAI API with the specified parameters."""
        response = openai.Completion.create(engine=self.get_engine(),
                                            prompt=self.craft_query(prompt),
                                            max_tokens=self.get_max_tokens(),
                                            temperature=self.get_temperature(),
                                            top_p=1,
                                            n=1,
                                            stream=False,
                                            stop=self.stop)
        return response

    def get_top_reply(self, prompt):
        """Obtains the best result as returned by the API."""
        response = self.submit_request(prompt)
        return response['choices'][0]['text']

    def format_example(self, ex):
        """Formats the input, output pair."""
        return self.input_prefix + ex.get_input(
        ) + self.input_suffix + self.output_prefix + ex.get_output(
        ) + self.output_suffix

    
    

class Sys:
    def list_files(self):
        import os
        files=os.listdir(".")
        print("FILES")
        for f in files:
            print(f)
    
    def get_dir(self):
        import os
        print(os.getcwd())
    
    def get_data(self):
        self.df = pd.read_csv('telec_data.csv')
        return self.df
    
    def data_checks(self, notebook=False):
        print('########## Performing Basic Data Checks ##########')
        print(f"Data dimensions: {self.df.shape}")
        print(f"Data Info: {self.df.info()}")
        print(f"Number of duplicates: {self.df.duplicated().sum()}")
        print('\n')    
        for col in self.df.columns:
            print(col)
            if (self.df[col].nunique() < 31 & notebook):
                print(self.df[col].value_counts(dropna=False))
            else: print("likely numeric, too many unique vals")
            
    def preprocessing(self):
        self.df2 = self.df.copy()
        # if small proportion of NAs drop them
        if self.df2.isna().sum().sum()/len(self.df2) < 0.05:
            self.df2.dropna(inplace=True)

        #remove cust ID column
        self.df2 = self.df2.iloc[:,1:].copy()

        #Convert the predictor variable in a binary numeric variable
        self.df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
        self.df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

        #Combine levels in columns
        self.df2['StreamingMovies'].replace(to_replace='No internet service', value='No', inplace=True)
        self.df2['StreamingTV'].replace(to_replace='No internet service', value='No', inplace=True)
        self.df2['TechSupport'].replace(to_replace='No internet service', value='No', inplace=True)
        self.df2['DeviceProtection'].replace(to_replace='No internet service', value='No', inplace=True)
        self.df2['OnlineBackup'].replace(to_replace='No internet service', value='No', inplace=True)
        self.df2['OnlineSecurity'].replace(to_replace='No internet service', value='No', inplace=True)
        self.df2['MultipleLines'].replace(to_replace='No phone service', value='No', inplace=True)

        # get cat cols
        cat = list(self.df2.select_dtypes(include='O').keys())
        for i in cat:
            self.df2[i] = self.df2[i].replace('Yes',1)
            self.df2[i] = self.df2[i].replace('No',0)

        # convert male = 1 and female = 0
        self.df2.gender = self.df2.gender.replace('Male',1)
        self.df2.gender = self.df2.gender.replace('Female',0)

        from sklearn.preprocessing import LabelEncoder
        label = LabelEncoder()
        self.df2['InternetService'] = label.fit_transform(self.df2['InternetService'].astype(str))
        self.df2['Contract'] = label.fit_transform(self.df2['Contract'].astype(str))
        self.df2['PaymentMethod'] = label.fit_transform(self.df2['PaymentMethod'].astype(str))

        # scale numeric cols
        scale_cols = ['tenure','MonthlyCharges','TotalCharges']
        self.df2['TotalCharges']=(self.df2['TotalCharges'].str.strip()).replace('',0).astype(float)

        from sklearn.preprocessing import MinMaxScaler
        scale = MinMaxScaler()
        self.df2[scale_cols] = scale.fit_transform((self.df2[scale_cols]))
        return self.df2


    
    def prep_train_test(self):
        self.x = self.df2.drop('Churn',axis=1)
        self.y = self.df2['Churn']

        from sklearn.model_selection import train_test_split
        self.xtrain,self.xtest,self.ytrain,self.ytest = train_test_split(self.x,self.y,test_size=0.2,random_state=10)

        return self.xtrain,self.xtest,self.ytrain,self.ytest

    def train_model(self):

        # define sequential model
        import tensorflow as tf
        from tensorflow import keras
        import pandas as pd, numpy as np

        tf.compat.v1.disable_eager_execution()

        self.model = keras.Sequential([
            # input layer
            keras.layers.Dense(19, input_shape=(19,), activation='relu'),
            keras.layers.Dense(15, activation='relu'),
            keras.layers.Dense(10,activation = 'relu'),
            # we use sigmoid for binary output
            # output layer
            keras.layers.Dense(1, activation='sigmoid')
        ]
        )

        # time for compilation of neural net.
        self.model.compile(optimizer = 'adam',
                     loss = 'binary_crossentropy',
                     metrics = ['accuracy'])
        # now we fit our model to training data
        self.model.fit(self.xtrain,self.ytrain,epochs=100)


        # predict the churn values
        self.ypred = self.model.predict(self.xtest)
        # unscaling the ypred values 
        self.ypred_lis = []
        for i in self.ypred:
            if i>0.5:
                self.ypred_lis.append(1)
            else:
                self.ypred_lis.append(0)

        #make dataframe for comparing the orignal and predict values
        self.data = {'orignal_churn':self.ytest, 'predicted_churn':self.ypred_lis}
        self.df_check = pd.DataFrame(self.data)
        print(self.df_check.head(10))

    def confusion_matrix(self):
        from sklearn.metrics import confusion_matrix, classification_report
        print(classification_report(self.ytest, self.ypred_lis))
        from sklearn.metrics import accuracy_score
        print(accuracy_score(self.ytest, self.ypred_lis, normalize=True))

        
        
    def gpt3_generate(self):
        import openai
        import os
        import sys
#         from api import GPT, Example
        openai.api_key = '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'

        # Construct GPT object and show some examples
        gpt = GPT(engine="davinci",
                  temperature=0.4,
                  max_tokens=100)
        
        def generate_email(userPrompt =""):
            """Returns a generated an email using GPT3 with a certain prompt and starting sentence"""

            response = openai.Completion.create(
            engine="davinci",
            prompt=userPrompt,
            temperature=0.71,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.36,
            presence_penalty=0.75
            )
            return response.get("choices")[0]['text']
        
        print("Generating online security blurbs for emails........")
        onlinesecurity1 = generate_email("Give me use cases for the importance of great online security?")
        onlinesecurity2 = generate_email("What are the features of great online security?")
        onlinesecurity3 = generate_email("Why is online security important?")
        onlinesecurity4 = generate_email("What is an advantage of having top tier online security?")
        onlinesecurity5 = generate_email("How to make the most out of online security?")
        data = {"Blurb#":[1,2,3,4,5],
                "Blurb":[onlinesecurity1,onlinesecurity2,onlinesecurity3,onlinesecurity4,onlinesecurity5]}
        gptdf = pd.DataFrame(data)
        gptdf.to_csv('Generate_OnlineSecurity_Text_Starter_Blurbs.csv')
        
        print("Generating device protection blurbs for emails........")
        deviceprotection1 = generate_email("Give me use cases for the importance of great device protection?")
        deviceprotection2 = generate_email("What are the features of great device protection?")
        deviceprotection3 = generate_email("Why is device protection important?")
        deviceprotection4 = generate_email("What is an advantage of having top tier device protection?")
        deviceprotection5 = generate_email("How to make the most out of device protection?")
        data = {"Blurb#":[1,2,3,4,5],
        "Blurb":[deviceprotection1,deviceprotection2,deviceprotection3,deviceprotection4,deviceprotection5]}
        gptdf = pd.DataFrame(data)
        gptdf.to_csv('Generate_deviceprotection_Text_Starter_Blurbs.csv')

        print("Generating internet service blurbs for emails........")
        serviceinternet1 = generate_email("Give me use cases for the importance of great service internet?")
        serviceinternet2 = generate_email("What are the features of great service internet?")
        serviceinternet3 = generate_email("Why is service internet important?")
        serviceinternet4 = generate_email("What is an advantage of having top tier service internet?")
        serviceinternet5 = generate_email("How to make the most out of service internet?")
        data = {"Blurb#":[1,2,3,4,5],
        "Blurb":[serviceinternet1,serviceinternet2,serviceinternet3,serviceinternet4,serviceinternet5]}
        gptdf = pd.DataFrame(data)
        gptdf.to_csv('Generate_serviceinternet_Text_Starter_Blurbs.csv')
        
        print("Generating tech support blurbs for emails........")
        techsupport1 = generate_email("Give me use cases for the importance of great tech support?")
        techsupport2 = generate_email("What are the features of great tech support?")
        techsupport3 = generate_email("Why is tech support important?")
        techsupport4 = generate_email("What is an advantage of having savvy tech support?")
        techsupport5 = generate_email("How to make the most out of tech support?")
        data = {"Blurb#":[1,2,3,4,5],
        "Blurb":[techsupport1,techsupport2,techsupport3,techsupport4,techsupport5]}
        gptdf = pd.DataFrame(data)
        gptdf.to_csv('Generate_techsupport_Text_Starter_Blurbs.csv')

    def generate_email_notebook(self, userPrompt ="write a letter"):
        """Returns a generated an email using GPT3 with a certain prompt and starting sentence.
        Main difference from function earlier is this truncates a shorter response which is feasible in notebook
        otherwise it takes so long"""
        openai.api_key = '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'
        response = openai.Completion.create(
        engine="davinci",
        prompt=userPrompt,
        temperature=0.71,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0.36,
        presence_penalty=0.75
        )
        return response.get("choices")[0]['text']
        

    
# localrun
if __name__ == "__main__":
    ch = Sys()
    df = ch.get_data()
    ch.data_checks()
    df2 = ch.preprocessing()
    xtrain,xtest,ytrain,ytest=ch.prep_train_test()
    ch.train_model()
    ch.confusion_matrix()
    ch.gpt3_generate()
    print("Finished Run")