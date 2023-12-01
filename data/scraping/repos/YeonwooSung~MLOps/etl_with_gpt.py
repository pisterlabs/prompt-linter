import openai

# set API key to use OpenAI API
openai.api_key = "YOUR KEY HERE"


def get_api_result(prompt):
    request = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[{"role": "user", "content": prompt}]
    )
    result = request['choices'][0]['message']['content']

    print(result)


def extract_with_gpt(prompt):
    prompt_template = """You are a ChatGPT language model that can generate Python code. Please provide a natural language input text, and I will generate the corresponding Python code.\nInput: {}\nPython code:""".format(prompt)
    get_api_result(prompt_template)


def transform_with_gpt(columns, column_types, prompt):
    prompt_template = """You are a ChatGPT language model that can generate Python code. Please provide a natural language input text, and I will generate the corresponding Python code using the Pandas to preprocess the DataFrame. The DataFrame columns are {} and their corresponding dtypes are {}.\nInput: {}\nPython code:""".format(columns, column_types, prompt)
    get_api_result(prompt_template)


def load_with_gpt(prompt):
    prompt_template = """You are a ChatGPT language model that can generate Python code. Please provide a natural language input text, and I will generate the corresponding Python code.\nInput: {}\nPython code:""".format(prompt)
    get_api_result(prompt_template)


def train_sklean_with_gpt(prompt):
    prompt_template = """You are a ChatGPT language model that can generate Python code. Focus on using scikit-learn when applicable. Please provide a natural language input text, and I will generate the corresponding Python code.\nInput: {}\nPython code:""".format(prompt)
    get_api_result(prompt_template)


def serve_model_on_mlflow_with_gpt(model_path, prompt):
    prompt_template = """You are a ChatGPT language model that can generate shell code for deploying models using MLFlow. Please provide a natural language input text, and I will generate the corresponding command to deploy the model. The model is located in the file {}.\nInput: {}\nShell command:""".format(model_path, prompt)
    get_api_result(prompt_template)


if __name__ == "__main__":
    extract_prompt = """Retrieve the adult income prediction dataset from openml using the sklearn fetch_openml function.Make sure to retrieve the data as a single dataframe which includes the target in a column named "target". Name the resulting dataframe "df"."""

    print('Extracting...')
    extract_with_gpt(extract_prompt)
    print('\n\n\n\n')

    print('Transforming...')
    transform_prompt = '''Preprocess the dataframe by converting all categorical columns to their one-hot encoded equivalents, and normalizing numerical columns. Drop rows which have an NA or NaN value in any column. Drop rows that have numeric column outliers as determined by their z score. A numeric column outlier is a value that is outside of the 1 to 99 inter-quantile range. The numerical columns should be normalized using StandardScaler from sklearn. The values in the target colummn should be converted to 0 or 1 and should be of type int.'''
    transform_with_gpt(transform_prompt)
    print('\n\n\n\n')

    print('Loading...')
    load_prompt = '''Connect to an sqlite database named "data". Use pandas to insert data from a DataFrame named "df" into a table named "income". Do not include the index column. Commit the changes before closing the connection.'''
    load_with_gpt(load_prompt)
    print('\n\n\n\n')

    print('Training...')
    train_prompt = '''Train a variety of classification models to predict the "target" column using all other columns. Do so using 5-fold cross validation to choose the best model and corresponding set of hyperparameters, and return the best overall model and corresponding hyperparameter settings. Choose the best model based on accuracy. Assume a dataframe named "df" exists which is to be used for training. Log the entire process using MLFlow. Start logging with mlflow before training any models so only a single run is stored. Make sure that the model is logged using the sklearn module of mlflow. Make sure that only the best overall model is logged, but log metrics for all model types. The mean value of the following metrics on all cross validation folds should be logged: accuracy, AUC, F1 score'''
    train_sklean_with_gpt(train_prompt)
    print('\n\n\n\n')

    print('Serving...')
    serve_prompt = '''Serve the model using port number 1111, and use the local environment manager'''
    serve_model_on_mlflow_with_gpt(serve_prompt)
