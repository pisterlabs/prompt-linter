import base64
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
import mlflow
from urllib.parse import urlparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import metrics as mt
import plotly.express as px
import streamlit as st
from PIL import Image
import altair as alt
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px
import pandas_profiling





# setting up the page streamlit

st.set_page_config(
    page_title="Linear Regression App ", layout="wide", page_icon="./images/linear-regression.png"
)


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()

#image_nyu = Image.open('images/nyu.png')
#st.image(image_nyu, width=100)




# navigation dropdown
st.sidebar.header("Dashboard")
st.sidebar.markdown("---")

#get model
model_mode = st.sidebar.selectbox('🔎 Select Model',['Linear Regression','Logistic Regression'])
    

# get pages
app_mode = st.sidebar.selectbox('📄 Select Page',['Introduction 🏃','Visualization 📊','Prediction 🌠','Deployment 🚀','Chatbot 🤖'])

#load data
#@st.cache_resource(experimental_allow_widgets=True)
def get_dataset(select_dataset):
    if "Spotify Top 50 🎼" in select_dataset:
        df = pd.read_csv("music.csv")
        df = df.dropna()
        return select_dataset, df


DATA_SELECT = {
    "Linear Regression": ["Spotify Top 50 🎼"],
    "Logistic Regression": ["Spotify Top 50 🎼"]
}

MODELS = {
    "Linear Regression": LinearRegression,
    "Logistic Regression": LogisticRegression 
}
target_variable = {
    "Spotify Top 50 🎼": "Danceability"
}

#image_header = Image.open('./images/Linear-Regression1.webp')
#st.image(image_header, width=600)
# page 1 
if app_mode == 'Introduction 🏃':
    if model_mode == 'Linear Regression':
        st.title("Linear Regression Lab 🧪")

#image_header = Image.open('./images/Logistic-Regression.jpg')
#st.image(image_header, width=600)
    elif model_mode == 'Logistic Regression':
        st.title("Logistic Regression Lab 🧪") 

    select_data =  st.sidebar.selectbox('💾 Select Dataset',DATA_SELECT[model_mode])
    select_dataset, df = get_dataset(select_data)

    st.markdown("### 00 - Show  Dataset")
    # Spotify top 50 dataset
    if select_dataset == "Spotify Top 50 🎼":
        
        st.markdown(
            """
            <style>
                div[data-testid="column"]:nth-of-type(1)
                {
                    border:1px solid blue;
                    text-align: center;
                    font-family: bariol;
                } 

                div[data-testid="column"]:nth-of-type(2)
                {
                    border:1px solid blue;
                    text-align: center;
                   
                }
                div[data-testid="column"]:nth-of-type(3)
                {
                    border:1px solid blue;
                    text-align: center;
                }
                div[data-testid="column"]:nth-of-type(4)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(5)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(6)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(7)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(8)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(9)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(10)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
            </style>
            """,unsafe_allow_html=True
        )
        col1, col2, col3,col4,col5,col6,col7,col8,col9,col10 = st.columns(10)
        col1.markdown(" **Country** ")
        col1.markdown("Country ISO code (3-letters)")
        col2.markdown(" **Track Name** ")
        col2.markdown("Name of the track")
        col3.markdown(" **Artist Name** ")
        col3.markdown("Name of the artist of the song")
        col4.markdown(" **Album Name** ")
        col4.markdown("Name of the album of the song")
        col5.markdown(" **Popularity** ")
        col5.markdown("A value assigned to define the popularity of each track. It can be any value between 0 (not popular) to 100 (really popular)")
        col6.markdown(" **Date** ")
        col6.markdown("Publish date of the song")
        col7.markdown(" **Markets** ")
        col7.markdown("First market where the song is available")
        col8.markdown(" **Danceability** ")
        col8.markdown("How suitable a track is for dancing based on a combination of musical elements.")
        col9.markdown(" **Acousticness** ")
        col9.markdown("How acoustic the song is")
        col10.markdown(" **Duration** ")
        col10.markdown("Duration in ms of the song")
        
        st.markdown(
            """
            <style>
                div[data-testid="column"]:nth-of-type(1)
                {
                    border:1px solid blue;
                    text-align: center;
                    font-family: bariol;
                } 

                div[data-testid="column"]:nth-of-type(2)
                {
                    border:1px solid blue;
                    text-align: center;
                   
                }
                div[data-testid="column"]:nth-of-type(3)
                {
                    border:1px solid blue;
                    text-align: center;
                }
                div[data-testid="column"]:nth-of-type(4)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(5)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(6)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(7)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(8)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(9)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
                div[data-testid="column"]:nth-of-type(10)
                {
                    border:1px solid blue;
                    text-align: center;
                } 
            </style>
            """,unsafe_allow_html=True
        )
        col11, col12, col13,col14,col15,col16,col17,col18,col19,col20 = st.columns(10)
        col11.markdown(" **Energy** ")
        col11.markdown("Perceptual measure of intensity and activity")
        col12.markdown(" **Instrumentalness** ")
        col12.markdown("Predicts whether a track contains no vocals")
        col13.markdown(" **Key** ")
        col13.markdown("0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1")
        col14.markdown(" **Liveness** ")
        col14.markdown("Detects the presence of an audience in the recording")
        col15.markdown(" **Loudness** ")
        col15.markdown("The overall loudness of a track in decibels (dB)")
        col16.markdown(" **Mode** ")
        col16.markdown("Mode indicates the modality (major or minor) of a track")
        col17.markdown(" **Speechiness** ")
        col17.markdown("Speechiness detects the presence of spoken words in a track")
        col18.markdown(" **Tempo** ")
        col18.markdown("The overall estimated tempo of a track in beats per minute (BPM)")
        col19.markdown(" **TSignature** ")
        col19.markdown("How many beats are in each bar (or measure)")
        col20.markdown(" **Positiveness** ")
        col20.markdown("Musical positiveness conveyed by a track")
        
        
    
    num = st.number_input('No. of Rows', 5, 10)
    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
    if head == 'Head':
        if st.button("Show Code"):
            code = '''df.head(num)'''
            st.code(code, language='python')
        st.dataframe(df.head(num))
    else:
        if st.button("Show Code"):
            code = '''df.tail(num)'''
            st.code(code, language='python')
        st.dataframe(df.tail(num))
    
    st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
    st.text('(Rows,Columns)')
    st.write(df.shape)


    st.markdown("### 01 - Description")
    if st.button("Show Describe Code"):
        code = '''df.describe()'''
        st.code(code, language='python')
    st.dataframe(df.describe())


    st.markdown("### 02 - Missing Values")
    st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")
    #df = df.dropna()
    dfnull = df.isnull().sum()/len(df)*100
    totalmiss = dfnull.sum().round(2)
    st.write("Percentage of total missing values:",totalmiss)
    if st.button("Show the Code"):
        code = ''' dfnull = df.isnull().sum()/len(df)*100 
totalmiss = dfnull.sum().round(2)
        '''
        st.code(code, language='python')
    st.write(dfnull)
    if totalmiss <= 30:
        st.success("Looks good as we have less then 30 percent of missing values.")
        st.image(
            "https://media.giphy.com/media/3ohzdIuqJoo8QdKlnW/giphy.gif",
            width=400,
        )
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")
        st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

    
    st.markdown("### 03 - Completeness")
    st.markdown(" Completeness is defined as the ratio of non-missing values to total records in dataset.") 
    # st.write("Total data length:", len(df))
    nonmissing = (df.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df),2)
    st.write("Completeness ratio:",completeness)
    if st.button("Show the ratio Code"):
        code = ''' nonmissing = (df.notnull().sum().round(2))
completeness= round(sum(nonmissing)/len(df),2)
        '''
        st.code(code, language='python')
    st.write(nonmissing)
    
    if completeness >= 0.80:
        st.success("Looks good as we have completeness ratio greater than 0.85.")
        st.image(
            "https://media.giphy.com/media/3ohzdIuqJoo8QdKlnW/giphy.gif",
            width=400,
        )
    else:
        st.success("Poor data quality due to low completeness ratio( less than 0.85).")


    st.markdown("### 04 - Complete Report")

    st.button("Generate Report")
    #if st.button("Generate Report"):
        # pr = df.profile_report()
        # export=pr.to_html()
        # st.download_button(label="Download Full Report", data=export,file_name='report.html')
        #st.markdown(pr.to_html(), unsafe_allow_html=True)
        #st.write(pr)
        # prof = pandas_profiling.ProfileReport(df, explorative=True, minimal=True)

        # output = prof.to_file('output.html', silent=False)
        
    
    


# page 2
if app_mode == 'Visualization 📊':
    st.markdown("# :violet[Visualization 📊]")
    #select_dataset =  st.sidebar.selectbox('💾 Select Dataset',DATA_SELECT[model_mode])
    #select_dataset, df = get_dataset(select_dataset)
    #df = get_dataset("music.csv")
    df = pd.read_csv("music.csv")
    list_variables = df.columns

    symbols = st.multiselect("Select two variables",list_variables,["Energy", "Danceability"] )
   

    tab1, tab2, tab3, tab4= st.tabs(["Bar Chart 📊","Line Chart 📈","Correlation ⛖","Pairplot"])  
    #tab1, tab2= st.tabs(["Line Chart","📈 Correlation"])    
    
    #tab1 in visualisation
    tab1.subheader("Bar Chart📊")
    tab1.write(" ")
    if tab1.button("Show Bar Chart Code"):
        code = '''bar_chart(data=df, x=symbols[0], y=symbols[1], use_container_width=True)'''
        tab1.code(code, language='python')

    
    tab1.write(" ")
    tab1.bar_chart(data=df, x=symbols[0], y=symbols[1], use_container_width=True)

    #tab2 in visualisation
    tab2.subheader("Line Chart📈")
    tab2.write(" ")
    if tab2.button("Show Line Chart Code"):
        code = '''st.line_chart(data=df, x=symbols[0],y=symbols[1], width=0, height=0, use_container_width=True)'''
        tab2.code(code, language='python')
        tab2.write(" ")
        tab2.line_chart(data=df, x=symbols[0],y=symbols[1], width=0, height=0, use_container_width=True)

    #tab3 
    tab3.subheader("Correlation Chart ⛖")
    tab3.write(" ")
    if tab3.button("Show Correlation Code"):
        code = '''sns.heatmap(df.corr(),cmap= sns.cubehelix_palette(8),annot = True, ax=ax)'''
        tab3.code(code, language='python')

    tab3.write(" ")
    fig3,ax = plt.subplots(figsize=(25, 25))
    df_numeric = df.select_dtypes(include=['number'])
    sns.heatmap(df_numeric.corr().corr(),cmap= sns.cubehelix_palette(8),annot = True, ax=ax)
    tab3.pyplot(fig3)
    # Compute a correlation matrix and convert to long-form
    #corr_mat = df.corr().stack().reset_index(name="correlation")
    # g = sns.relplot(
    #     data=corr_mat,
    #     x="level_0", y="level_1", hue="correlation", size="correlation",
    #     palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    #     height=14, sizes=(20, 200), size_norm=(-.2, .8))


    tab4.subheader("pairplot Chart 🗠")
    tab4.write(" ")
    if tab4.button("Show pairplot Chart Code"): 
        code = '''sns.pairplot(df2)'''
        tab4.code(code, language='python')

    if tab4.button('show pairplot visualisation'):
        progress_text = "visualisation in progress🛞!!   :red[Please wait🛑]"
        my_bar = st.progress(0, text=progress_text)
        time.sleep(2)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(3)
        my_bar.empty()
        df2 = df[[list_variables[0],list_variables[1],list_variables[2],list_variables[3],list_variables[4]]]
        fig4 = sns.pairplot(df2.sample(500))
        tab4.write(" ")
        tab4.pyplot(fig4)



# page 3
if app_mode == 'Prediction 🌠':
    st.markdown("# :violet[Prediction 🌠]")
    select_ds =  st.sidebar.selectbox('💾 Select Dataset',DATA_SELECT[model_mode])
    select_dataset, df = get_dataset(select_ds)
    list_variables = target_variable[select_ds]

    if model_mode == 'Linear Regression':
        st.title("Linear Regression Lab 🧪")
        df = df.drop(['Country','Track Name','Artist Name','Album Name','Date','Markets'],axis=1)
        if st.button("Show ML Code 👀"):
            code = '''X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=train_size)'''
            code1= '''lm = LinearRegression()
            lm.fit(X_train,y_train)'''
            code2 = '''predictions = lm.predict(X_test)'''
            st.code(code, language='python')
            st.code(code1, language='python')
            st.code(code2, language='python')
    elif model_mode == 'Logistic Regression':
        st.title("Logistic Regression Lab 🧪")
        df = df.drop(['Popularity','Date','Markets','Danceability','Acousticness','duration', 'Energy', 'Instrumentalness', 'Key', 'Liveness', 'Loudness', 'Mode', 'Speechiness', 'Tempo', 'TSignature', 'Positiveness'],axis=1)
        #df = df.drop(['Country','Track Name','Artist Name','Album Name','Date','Markets'],axis=1)
        if st.button("Show ML Code 👀"):
            code = '''X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=train_size)'''
            code1= '''lm = LogisticRegression()
            lm.fit(X_train,y_train)'''
            code2 = '''predictions = lm.predict(X_test)'''
            st.code(code, language='python')
            st.code(code1, language='python')
            st.code(code2, language='python')
        
    # converting data
    # if select_dataset == "Student Score 💯":
    #     # Use apply with a lambda function to map values
    #     df['Extracurricular Activities'] = df['Extracurricular Activities'].apply(lambda x: 1 if x == 'Yes' else 0)
        
    # elif select_dataset == "Income 💵":
    #     df = pd.get_dummies(df, columns=['education'], drop_first=True)
    #     df = df.drop(['workclass','occupation','education.num','relationship','race','native.country'],axis=1)
    #     columns_to_dummy = ['marital.status', 'sex']
    #     df = pd.get_dummies(df, columns=columns_to_dummy, drop_first=True)
    #     std = StandardScaler()
    #     mms = MinMaxScaler()
    #     columns_to_scaler = ['capital.gain', 'capital.loss', 'hours.per.week']
    #     df[columns_to_scaler] = std.fit_transform(df[columns_to_scaler]) 
    #     income_map = {'<=50K': 1, '>50K': 0}
    #     df['income'] = df['income'].map(income_map)
    
    # elif select_dataset == "Titanic 🛳️":
    #     columns_to_dummy = ['embarked', 'sex','class','alive']
    #     df = pd.get_dummies(df, columns=columns_to_dummy, drop_first=True)
    #     df = df.drop('adult_male',axis=1)


    #choose the dependent variable
    target_choice1 =  st.sidebar.selectbox('🎯 Select Variable to Predict',[list_variables,""])
    target_choice=list_variables
    st.experimental_set_query_params(saved_target=target_choice)
    
    #dropping the target column
    new_df= df.drop(labels=target_choice, axis=1)  #axis=1 means we drop data by columns
    #imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    #imp_mean.fit()

    #other column list
    list_var = new_df.columns
    feature_choice = st.multiselect("Select Explanatory Variables", list_var)
    train_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)
    
#     if st.button("Show ML Code 👀"):
#         code = '''X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=train_size)'''
#         code1= '''lm = LinearRegression()
# lm.fit(X_train,y_train)'''
#         code2 = '''predictions = lm.predict(X_test)'''
#         st.code(code, language='python')
#         st.code(code1, language='python')
#         st.code(code2, language='python')
    
    @st.cache_resource
    def predict(target_choice,train_size, new_df,feature_choice):
        #independent variables / explanatory variables
        #choosing column for target
        new_df2 = new_df[feature_choice]
        x =  new_df2
        y = df[target_choice]
        col1,col2 = st.columns(2)
        col1.subheader("Feature Columns top 25")
        col1.write(x.head(25))
        col2.subheader("Target Column top 25")
        col2.write(y.head(25))
        #X = df[feature_choice].copy()
        #y = df['target'].copy()
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_size)
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=train_size)
        lm = MODELS[model_mode]()
        model = lm.fit(X_train,y_train)
        predictions = lm.predict(X_test)
        return lm,X_train,y_test,predictions,model

    # Mlflow tracking
    track_with_mlflow = st.checkbox("Track with mlflow? 🛤️")

    # Model training
    start_training = st.button("Start training")
    if not start_training:
        st.stop()

    if mlflow.active_run():
        mlflow.end_run()
    if track_with_mlflow:
        #mlflow.set_tracking_uri("./model_metrics")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        experiment_name = select_dataset
        st.write(experiment_name)
        mlflow.start_run()
        try:
            # creating a new experiment
            exp_id = mlflow.create_experiment(name=experiment_name)
        except Exception as e:
            exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        #mlflow.set_experiment(select_dataset)
        mlflow.log_param('model', MODELS[model_mode])
        mlflow.log_param('features', feature_choice)
    
    lm,X_train,y_test,predictions,model = predict(target_choice,train_size,new_df,feature_choice)

    
    # Model evaluation
    #preds_train = model.predict(X_train)
    #preds_test = model.predict(X_test)
    #preds_test = lm.predict(X_test)
    # if problem_type=="classification":
    #     st.subheader('🎯 Results')
    #     metric_name = "f1_score"
    #     metric_train = f1_score(y_train, preds_train, average='micro')
    #     metric_test = f1_score(y_test, preds_test, average='micro')
    # else:
    #     st.subheader('🎯 Results')
    mae = np.round(mt.mean_absolute_error(y_test, predictions ),2)
    mse = np.round(mt.mean_squared_error(y_test, predictions),2)
    r2 = np.round(mt.r2_score(y_test, predictions),2)
    
    #metric_name = "r2_score"
    #metric_test = r2_score(y_test, preds_test)
    #st.write(metric_name+"_train", round(metric_train, 3))
    #st.write(metric_name+"_test", round(metric_test, 3))
    if track_with_mlflow:
       # mlflow.sklearn.log_model(lm, "top_model_v1")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.end_run()
    

    # Save the model to a PKL file
    with open('model.pkl', 'wb') as file:
        pickle.dump(lm, file)

    # model_code = st.checkbox("See the model code? 👀")
    # if model_code:
    #     code = '''X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=train_size)'''
    #     code1 = '''lm = LinearRegression()'''
    #     code2 = '''lm.fit(X_train,y_train)'''
    #     code3 = '''predictions = lm.predict(X_test)'''
    #     st.code(code, language='python')
    #     st.code(code1, language='python')
    #     st.code(code2, language='python')
    #     st.code(code3, language='python')
    
    st.subheader('🎯 Results')
    if model_mode == 'Linear Regression':
        st.write("1) The model explains,", np.round(mt.explained_variance_score(y_test, predictions)*100,2),"% variance of the target feature")
        st.write("2) The Mean Absolute Error of model is:", np.round(mae,2))
        st.write("3) MSE: ", np.round(mse))
        st.write("4) The R-Square score of the model is " , np.round(r2))
    else:
        acc = accuracy_score(y_test, predictions)
        st.write("1) Model Accuracy (in %):", np.round(acc*100,2))
        f1_score = f1_score(y_test, predictions, average='weighted')
        st.write("2) Model F1 Score (in %):", np.round(f1_score*100,2))
        precision_score = precision_score(y_test, predictions, average='weighted')
        st.write("3) Model Precision Score (in %):", np.round(precision_score*100,2))
        recall_score = recall_score(y_test, predictions, average='weighted')
        st.write("4) Model Recall Score (in %):", np.round(recall_score*100,2))

    @st.cache_resource
    def download_file():
        file_path = 'model.pkl'  # Replace with the actual path to your model.pkl file
        with open(file_path, 'rb') as file:
            contents = file.read()
        b64 = base64.b64encode(contents).decode()
        href = f'<a href="data:file/pkl;base64,{b64}" download="model.pkl">Download model.pkl file</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.title("Download Model Example")
        st.write("Click the button below to download the model.pkl file.")
    if st.button("Download"):
        download_file()
       # return X_train, X_test, y_train, y_test, predictions,x,y, mae,mse, r2

# import streamlit.components.v1 as components
# def st_shap(plot, height=None):
#     shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
#     components.html(shap_html, height=height)
# if app_mode == 'shap':
#     st.subheader('Result Interpretability - Applicant Level')
#     shap.initjs()
#     lm,X_train,y_test,predictions,model = predict(target_choice,train_size, new_df,feature_choice)
#     explainer = shap.Explainer(model) 
#     shap_values = explainer(X_train) 

#     explainer1 = shap.TreeExplainer(model)
#     shap_values1 = explainer1.shap_values(X_train)
#     # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
#     st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:]))

#     #visualize the training set predictions
#     st_shap(shap.force_plot(explainer.expected_value, shap_values, X_train), 400)
#     fig = shap.plots.bar(shap_values[0]) 
#     st.pyplot(fig) 

#     st.subheader('Model Interpretability - Overall') 
#     shap_values_ttl = explainer(X_train) 
#     fig_ttl = shap.plots.beeswarm(shap_values_ttl)
#     st.pyplot(fig_ttl) 

#page 5
if app_mode == 'Deployment 🚀':
    st.markdown("# :violet[Deployment 🚀]")
    select_ds =  "Wine Quality 🍷"
#    select_dataset, df = get_dataset(select_ds)

    id = st.text_input('ID Model', '00ffae4993044a5d9cb369a46dbc1e01')
        # Print emissions
    #logged_model = f'runs:/{id}/top_model_v1'
    logged_model = f'./mlruns/0/{id}/artifacts/top_model_v1'
    
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # app_state = st.experimental_get_query_params()  
    # #saved_target
    # if "saved_target" in app_state:
    #     target_choice = app_state["saved_target"][0]
    # else:
    #     st.write("target not saved")
    
    df = pd.read_csv("music.csv")
    deploy_df= df.drop(labels='fixed acidity', axis=1) 
    list_var = deploy_df.columns
    #st.write(target_choice)
    
    number1 = st.number_input(deploy_df.columns[0],0.7)
    number2 = st.number_input(deploy_df.columns[1],0.04)
    number3 = st.number_input(deploy_df.columns[2],1.1)
    number4 = st.number_input(deploy_df.columns[3],0.05)
    number5 = st.number_input(deploy_df.columns[4],25)
    number6 = st.number_input(deploy_df.columns[5],20)
    number7 = st.number_input(deploy_df.columns[6],0.98)
    number8 = st.number_input(deploy_df.columns[7],1.9)
    number9 = st.number_input(deploy_df.columns[8],0.4)
    number10 = st.number_input(deploy_df.columns[9],9.4)
    number11 = st.number_input(deploy_df.columns[10],5)

    data_new = pd.DataFrame({deploy_df.columns[0]:[number1], deploy_df.columns[1]:[number2], deploy_df.columns[2]:[number3],
         deploy_df.columns[3]:[number4], deploy_df.columns[4]:[number5], deploy_df.columns[5]:[number6], deploy_df.columns[6]:[number7],
         deploy_df.columns[7]:[number8], deploy_df.columns[8]:[number9],deploy_df.columns[9]:[number10],deploy_df.columns[10]:[number11]})
    # Predict on a Pandas DataFrame.
    #import pandas as pd
    st.write("Prediction :", np.round(loaded_model.predict(data_new)[0],2))

from streamlit_chat import message
import openai

#page 6
if app_mode == 'Chatbot 🤖':
    st.markdown("# :violet[ Your Personal Chatbot 🤖]")
   # OPENAI_API_KEY = "YOUR_API_KEY"
    # Set org ID and API key
    openai.organization = st.secrets.op_ai.org_key
    openai.api_key = st.secrets.op_ai.api_key

    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = []
    if 'cost' not in st.session_state:
        st.session_state['cost'] = []
    if 'total_tokens' not in st.session_state:
        st.session_state['total_tokens'] = []
    if 'total_cost' not in st.session_state:
        st.session_state['total_cost'] = 0.0

    # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
    model_name = "GPT-3.5"
    counter_placeholder = st.sidebar.empty()
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    clear_button = st.sidebar.button("Clear Conversation", key="clear")

    # Map model names to OpenAI model IDs
    if model_name == "GPT-3.5":
        model = "gpt-3.5-turbo"
    else:
        model = "gpt-4"

    # reset everything
    if clear_button:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.session_state['number_tokens'] = []
        st.session_state['model_name'] = []
        st.session_state['cost'] = []
        st.session_state['total_cost'] = 0.0
        st.session_state['total_tokens'] = []
        counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


    # generate a response
    def generate_response(prompt):
        st.session_state['messages'].append({"role": "user", "content": prompt})

        completion = openai.ChatCompletion.create(
            model=model,
            messages=st.session_state['messages']
        )
        response = completion.choices[0].message.content
        st.session_state['messages'].append({"role": "assistant", "content": response})

        # print(st.session_state['messages'])
        total_tokens = completion.usage.total_tokens
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        return response, total_tokens, prompt_tokens, completion_tokens


    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['model_name'].append(model_name)
            st.session_state['total_tokens'].append(total_tokens)

            # from https://openai.com/pricing#language-models
            if model_name == "GPT-3.5":
                cost = total_tokens * 0.002 / 1000
            else:
                cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

            st.session_state['cost'].append(cost)
            st.session_state['total_cost'] += cost

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                st.write(
                    f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
                counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


if __name__=='__main__':
    main()

st.markdown(" ")
st.markdown("### 👨🏼‍💻 **App Contributors:** ")
st.markdown("Nina Sukonrat, Yinyi Feng, Sayuri Hadge")
st.markdown(f"####  Link to Project Website [here]({'https://github.com/sayuh07/SpotifyTop50'}) 🚀 ")


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;background - color: white}
     .stApp { bottom: 80px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,

    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

