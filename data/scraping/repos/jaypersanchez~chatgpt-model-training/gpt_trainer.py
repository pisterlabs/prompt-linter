import os
from dotenv import load_dotenv
load_dotenv()
import openai
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit, QDesktopWidget, QTableView, QPushButton, QTabWidget, QFileDialog
from PyQt5.QtGui import QColor, QPixmap
#import requests
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import pickle
from PyQt5.QtCore import QAbstractTableModel, Qt, QRect
import tiktoken
from openai.embeddings_utils import get_embedding
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")


openai.organization = os.environ.get("ORG_ID")
openai.api_key =  os.environ.get("API_KEY")



class Satoshi(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('GPT Data Modeler')

        # Create tabs
        container = QTabWidget(self)
        tabs = QTabWidget(self)
        training_tab = QWidget()
        data_scaling_tab = QWidget()
        tabs.addTab(training_tab, 'Data Model Training')
        tabs.addTab(data_scaling_tab, "AWS Fine Food Review Embedding")
        training_tab.showMaximized()
        data_scaling_tab.showMaximized()
        #tabs.showMaximized() # Unable to maximize each tab

        # Validate data model coins
        self.button_validate_model = QPushButton('Validate Model')
        self.button_validate_model.clicked.connect(self.validate_model)

        # Create data model build file
        self.button_train_gpt = QPushButton('Start GPT Training')
        self.button_train_gpt.clicked.connect(self.train_gpt)
        
        # Exit application
        self.button_exit = QPushButton('Exit')
        self.button_exit.clicked.connect(self.close_application)

        # Create a label to display the message. Used for any error messages
        self.label = QLabel('')
        color = QColor(255,0,0) # red color
        self.label.setStyleSheet("color: {}".format(color.name()))

        # data modelling 
        self.data_textarea = QTextEdit()
        self.data_textarea.setOverwriteMode(True)
        self.data_textarea.toHtml()
        self.data_textarea.setPlaceholderText("Design Data Model in here")

        # status console
        self.console_textarea = QTextEdit()
        self.console_textarea.setOverwriteMode(True)
        self.console_textarea.toHtml()
        self.console_textarea.setPlaceholderText("Debug Output")

        # Create a layout to organize the UI elements
        trainingtab = QVBoxLayout(training_tab) # vertical
        training_tab.setLayout(trainingtab)
        training_tab.setGeometry(QRect(0, 0, self.width(), self.height()))
        datascalingtab = QVBoxLayout(data_scaling_tab)
        data_scaling_tab.setLayout(datascalingtab)
        data_scaling_tab.setGeometry(QRect(0, 0, self.width(), self.height()))
        horizontal_container = QHBoxLayout() # horizontal

        # layout items for Data Model Training
        trainingtab.addWidget(self.button_validate_model)
        trainingtab.addWidget(self.button_train_gpt)
        self.button_validate_model.setDisabled(True)
        self.button_train_gpt.setDisabled(True)
        trainingtab.addLayout(horizontal_container)
        horizontal_container.addWidget(self.data_textarea)
        horizontal_container.addWidget(self.console_textarea)
        trainingtab.setGeometry(QRect(0, 0, self.width(), self.height()))

        # layout items for Data Regression Model
        self.button_import_data_file = QPushButton('Food Recommendations')
        self.button_import_data_file.clicked.connect(self.embeddedRecommendations)
        self.button_Food_Ratings = QPushButton("Food Ratings Visual View")
        self.button_Food_Ratings.clicked.connect(self.plotFoodRatings)
        datascalingtab.addWidget(self.button_import_data_file)
        datascalingtab.addWidget(self.button_Food_Ratings)

    def plotFoodRatings(self):
        fileName, _ = QFileDialog.getOpenFileName(None, "Select File", "", "All Files (*);;Python Files (*.py)")
        colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
        if fileName:
            #print(fileName)
            df = pd.read_csv(fileName)
            matrix = np.array(df.embedding.apply(eval).to_list())
            tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
            vis_dims = tsne.fit_transform(matrix)
            vis_dims.shape
            x = [x for x,y in vis_dims]
            y = [y for x,y in vis_dims]
            color_indices = df.Score.values - 1
            colormap = matplotlib.colors.ListedColormap(colors)
            plt.scatter(x,y, c=color_indices, cmap=colormap, alpha=3)
            for score in [0,1,2,3,4]:
                avg_x = np.array(x)[df.Score-1==score].mean()
                avg_y = np.array(y)[df.Score-1==score].mean()
                color = colors[score]
                plt.scatter(avg_x, avg_y, marker='x', color=color, s=100)
            plt.title("Amazon ratings visualized in language using t-SNE")

    def embeddedRecommendations(self):
        fileName, _ = QFileDialog.getOpenFileName(None, "Select File", "", "All Files (*);;Python Files (*.py)")
        
        embedding_cache_path = "./models/recommendations_embeddings_cache.pkl"
        try: 
            embedding_cache = pd.read_pickle(embedding_cache_path)
        except FileNotFoundError:
            embedding_cache = {}
        with open(embedding_cache_path, "wb") as embedding_cache_file:
                pickle.dump(embedding_cache, embedding_cache_file)

        if fileName:
            #print(fileName)
            df = pd.read_csv(fileName)
            #n_examples = 5
            #df.head(n_examples)
            df_summary = df['Summary'].values[0]
            if(df_summary, EMBEDDING_MODEL) not in embedding_cache.keys():
                embedding_cache[(df_summary, EMBEDDING_MODEL)] = get_embedding(df_summary, EMBEDDING_MODEL)
            with open(embedding_cache_path, "wb") as embedding_cache_file:
                pickle.dump(embedding_cache, embedding_cache_file)
            output_embedding = embedding_cache[(df_summary, EMBEDDING_MODEL)]
            print(f"\nSummary Values: {df['Summary'].values[0]}")
            #print(f"\nOutput to be embedded: {output_embedding}")
            print(f"\nEmbedded Values: {output_embedding[:10]}")

    def RegressionScaling(self):
        fileName, _ = QFileDialog.getOpenFileName(None, "Select File", "", "All Files (*);;Python Files (*.py)")
        if fileName:
            print(fileName)
            dataset = pd.read_csv(fileName)
            X = dataset.iloc[:,:-1].values # independent variable
            # dependent variable. the one we want to answer a question
            Y = dataset.iloc[:, -1] # -1 index of last column

            # clean up data like
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, 1:3])
            X[:, 1:3] = imputer.transform(X[:, 1:3])

            # Categorical data
            

            #print
            print(X)
            print(Y)

    # Send prompt to GPT
    def validate_model(self):
        print('Validate Model')
        model = "text-davinci-002"
        '''
        train_data = ["This is an example training sentence.", "Another example sentence for training."]
        validation_data = ["This is an example validation sentence.", "Another example validation sentence."]
        '''
        train_data = "<READ FROM MODEL FILE>" #prompt
        validation_data = "" #complete
        config = {
            "epochs": 3,
            "batch_size": 2,
            "learning_rate": 1e-5,
            "early_stopping": True,
            "validation_split": 0.1
        }
        fine_tune = openai.FineTune.create(model=model, train_data=train_data, validation_data=validation_data, **config)
        # track fine tune status
        run_id = fine_tune["id"]
        status = openai.FineTune.retrieve(run_id)["status"]
        print(f"Fine-tuning run {run_id} is {status}")


    # Train GPT
    def train_gpt(self):
        fine_tuned_model = f"{model}-{run_id}"
        completion = openai.Completion(engine=fine_tuned_model)
        print('Start GPT Training on selected data model')

    # Exit application
    def close_application(self):
        window.close()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Satoshi()
    #screen = QDesktopWidget().screenGeometry()
    window.setGeometry(QDesktopWidget().screenGeometry())
    window.show()
    sys.exit(app.exec_())


