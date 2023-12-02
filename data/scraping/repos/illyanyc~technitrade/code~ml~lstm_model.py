'''Machine Leaning Model Class'''

# Importing Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import RobustScaler

# Neural Network library
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model


class MachineLearningModel:
    '''
    Class used to handle the following LSTM tasks:
    1. Data pre-processing :  Scaling Data
    2. Model building : LSTM hidden layer structuring
    3. Model training : Training model on technical indicator data
    4. Model validation : Validating model
    5. Exporting and saving the model : Saving as .h5 file recommended
    
    Functions
    ---------
    
    print_df() : prints the dataframe used to instantiate the class
    
    split_sequence() : splits the multivariate time sequence
    
    '''
    
    def __init__(self, 
                 data : pd.DataFrame,
                 n_in : int = 100,
                 n_out : int = 14,
                 n_layers : int = 1,
                 n_nodes : int = 30,
                 epochs : int = 16,
                 batch_size : int = 128,
                 validation_split : float = 0.1,
                 activation : str = "tanh",
                 optimizer : str ='adam', 
                 loss : str ='mse'):
        '''
        Parameters
        ----------
        data : DataFrame
            data consisting of technical indicators and market data
        
        n_in : int
            number of periods looking back to learn
            default = 100
            
        n_out : int
            number of periods to predict
            default = 30
            
        n_layers : int
            number of hidden layers in add_layer() class method
            will build n number of hidden layers for the model
            default = 1
            
        n_nodes : int
            number of nodes in each layer built by the add_layer() method
            each layer built by the above-mentioned method will contain n nodes
            default = 30
            
        epochs : int
            number of epochs for LSTM training
            default = 50
            
        batch_size : int
            batch size for LSTM trainig, number of data-items per epoch
            default = 128
            
        validation_split : float
            amount of data to be used for model validation during training
            default = 0.1
            
        activateion : str
            activation method used by the LSTM model
            default = "tanh" 
                tanh : Sigmoid specifically, is used as the gating function for the three gates (in, out, and forget) in LSTM, since it outputs a value between 0 and 1, and it can either let no flow or complete flow of information throughout the gates.
            Full list of all activation functions: 
                https://www.tensorflow.org/api_docs/python/tf/keras/activations
                
        optimizer : str
            optimzer used by LSTM model
            default = "adam"
                adam: Acronynm for "adaptive moment estimation". 
                Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data. Adam was presented by Diederik Kingma from OpenAI and Jimmy Ba from the University of Toronto in their 2015 ICLR paper (poster) titled “Adam: A Method for Stochastic Optimization“. I will quote liberally from their paper in this post, unless stated otherwise.
            Full list of all optimizers:
                https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
                
        loss : str
            loss function used by the LSTM model
            default = mse
                mse : Acronym for "mean squared error".
                MSE is sensitive towards outliers and given several examples with the same input feature values, the optimal prediction will be their mean target value. 
        '''
        
        self.df = data
        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        
        # LSTM class variables
        self.n_features = None
        self.close_scaler = None
        self.model = None
        self.train_df = None
        self.predictions = None
        self.rmse_value = None
        

    ########################
    ### Helper Functions ###
    ########################
    def print_df(self):
        '''Prints DataFrame head'''
        print(self.df.head())
        
    def get_model_summary(self):
        '''Prints Model Summary'''
        try:
            self.model.summary()
        except:
            print("Model is not built.")
            
    def get_model(self):
        '''Returns Model Object'''
        return self.model
    
    def drop_columns(self, cols : list = ['open', 'high', 'low', 'volume']):
        '''Drops pd.DataFrame colums'''
        try:
            self.df.drop(columns=cols, inplace=True)
        except:
            print("Dataframe un-used columns alread dropped")
        
    def set_model_shape(self):
        '''Sets model shape by passing shape to n_features'''
        self.n_features = self.df.shape[1]
    
    
    
    #################################
    ### LSTM Model Data Functions ###
    #################################
    def split_sequence(self, sequence):
        '''
        Splits the multivariate time sequence
        
        Parameters
        ----------
        sequence : np.array
            numpy array of the dataframe used to train the model
        
        Returns
        -------
        X, y : np.array
            Time sequence values for X and y portions of the dataset
        '''

        # creating a list for both variables
        X, y = [], []

        for i in range(len(sequence)):

            # finding the end of the current sequence
            end = i + self.n_in
            out_end = end + self.n_out

            # breaking out of the loop if we have exceeded the dataset length
            if out_end > len(sequence):
                break

            # splitting the sequences into: x = past prices and indicators, y = prices ahead
            sequence_x, sequence_y = sequence[i:end, :], sequence[end:out_end, 0]

            X.append(sequence_x)
            y.append(sequence_y)

        return np.array(X), np.array(y)

    
    def add_hidden_layers(self, 
                          n_layers : int, 
                          n_nodes : int, 
                          activation : int, 
                          drop : int = None, 
                          drop_rate : float = 0.5):
        '''
        Creates a specific amount of hidden layers for the model
        
        Parameters
        ----------
        n_layers : int
            number of layers to be added to the model
            
        n_nodes : int
            number of nodes to be added to each layer
            
        activation : str
            activation function used by each layers in the model
            Full list of all activation functions: 
                https://www.tensorflow.org/api_docs/python/tf/keras/activations
        
        drop : int
            every n-th hidden layer after which a Dropout layer to be added
        
        drop_rate : float
            rate for each Dropout layer
            default = 0.5
        
        '''

        # creating the specified number of hidden layers with the specified number of nodes
        for x in range(1,n_layers+1):
            self.model.add(LSTM(n_nodes, activation=activation, return_sequences=True))

            # adds a Dropout layer after every n-th hidden layer
            try:
                if x % drop == 0:
                    self.model.add(Dropout(drop_rate))
            except:
                pass
            
    def add_dense_layers(self, n_layers : int, n_out : int):
        '''
        Creates a specific amount of Dense layers for the model
        
        Parameters
        ----------
        n_layers : int
            number of layers to be added to the model    
        '''

        # creating the specified number of hidden layers with the specified number of nodes
        for x in range(1,n_layers+1):
            self.model.add(Dense(n_out))
            
    def validate(self):
        '''Vaildates predictions'''
        self.predictions = self.validater()
        self.rmse()

        
    def validater(self):
        '''
        Creates predicted values.
        
        Returns
        -------
        predictions : pd.DataFrame
            Predicted values for the model
        '''
        
        # create empty pd.DataFrame to store predictions
        predictions = pd.DataFrame(index=self.train_df.index, columns=[self.train_df.columns[0]])

        for i in range(self.n_in, len(self.train_df)-self.n_in, self.n_out):
            # create data time windows
            x = self.train_df[-i - self.n_in:-i]
            # predict using the time window
            y_pred = self.model.predict(np.array(x).reshape(1, self.n_in, self.n_features))
            
            # inverse the close scaler to return 'close' values
            y_pred = self.close_scaler.inverse_transform(y_pred)[0]
            
            # store values and append using business-days as frequency
            pred_df = pd.DataFrame(y_pred, 
                                   index=pd.date_range(start=x.index[-1], 
                                                       periods=len(y_pred), 
                                                       freq="B"),
                                   columns=[x.columns[0]])
            
            # Updating the predictions DF
            predictions.update(pred_df)
        
        predictions = predictions.fillna(method='bfill')

        return predictions


    def rmse(self):
        '''
        Calculates the RMS (root mean square) error between the two pd.Dataframes
        '''
        df = pd.DataFrame(self.df['close'].copy())
        df['close_pred'] = self.predictions
        df.dropna(inplace=True)
        df['diff'] = df['close'] - df['close_pred']
        rms = (df[['diff']]**2).mean()
        error = float(np.sqrt(rms))
        self.rmse_value = error

    
    
    ######################################
    ### LSTM Model Builder and Trainer ###
    ######################################
    def build_model(self, summary : int = 1, verbose : int = 0):
        '''
        Trains LSTM model : 
            1. Scales the data using RobustScaler()
                Scale features using statistics that are robust to outliers. This Scaler removes the median and scales the data according to the quantile range. Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Median and interquartile range are then stored to be used on later data using the transform method. 
                Ref : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
                
            2. Splits the sequence into X and y : self.split_sequence()
            
            3. Builds LSTM model : hard-coded layers and self.add_hidden_layers()
            
            4. Trains LSTM Model
            
        Returns
        -------
        trained_model : tf.model
            Trained LSTM model history
        '''
        # drop un-used columns from pd.DataFrame
        self.drop_columns()
        
        # set self.n_features parameter
        self.set_model_shape()
        
        # deep copy the pd.DataFrame containing technical indicators
        self.train_df = self.df.copy(deep=True)

        # declare a scaler using RobustScaler() for 'close' data
        self.close_scaler = RobustScaler()
        self.close_scaler.fit(self.train_df[['close']])
    
        # declare a scaler using RobustScaler() for technical indicator data
        scaler = RobustScaler()
        
        # scale the data
        self.train_df = pd.DataFrame(scaler.fit_transform(self.train_df), columns=self.train_df.columns, index=self.train_df.index)
        
        # split data into appropriate sequences
        X, y = self.split_sequence(self.train_df.to_numpy())
        
        # instatiate the TensorFlow model
        self.model = Sequential()

        # create an input layer
        self.model.add(LSTM(90, 
                       activation=self.activation, 
                       return_sequences=True, 
                       input_shape=(self.n_in, self.n_features)))

        # add hidden layers
        self.add_hidden_layers(n_layers=self.n_layers, 
                               n_nodes=self.n_nodes, 
                               activation=self.activation)

        # add the last hidden layer
        self.model.add(LSTM(60, activation=self.activation))

        # add output layers
        self.add_dense_layers(n_layers=1, n_out=30)
        self.add_dense_layers(n_layers=1, n_out=self.n_out)

        # compile the data
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        
        if summary == 1:
            self.model.summary()

        hist = self.model.fit(X, y, 
                              epochs=self.epochs, 
                              batch_size=self.batch_size,
                              validation_split=self.validation_split, 
                              verbose=verbose)

        return hist

    
    #####################################
    ### Model export/import functions ###
    #####################################
    def save_model(self, filename : str, filetype : str = 'h5'):
        '''Saves model'''
        
        if filetype == 'h5':
            '''Saves the entire model'''
            self.model.save(filename+'.h5')
            
        elif filetype == 'json':
            '''Saves only model architecture'''
            string = self.model.to_json()
            return string
        
        elif filetype == 'weights':
            '''Saves model weights'''
            self.model.save_weights(filepath+'.h5')
            
        else:
            print("Incorrect model file type.")
            
        
    def load_model(self, filename : str):
        '''Loads model'''
        self.model = load_model(filename)
        
    ###############################
    ### Visualization Functions ###
    ###############################
    def visualize_training_results(self, hist):
        '''
        Visualizes the training results  
        '''
        
        # plot
        history = hist.history
        plt.figure(figsize=(16,5))
        plt.plot(history['val_loss'])
        plt.plot(history['loss'])
        plt.legend(['val_loss', 'loss'])
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        
        plt.figure(figsize=(16,5))
        plt.plot(history['val_accuracy'])
        plt.plot(history['accuracy'])
        plt.legend(['val_accuracy', 'accuracy'])
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()

        
    def visualize_training_price(self):
        '''
        Visualizes Actual vs. Predicted stock price
        '''

        # plot
        plt.figure(figsize=(16,6))
        plt.plot(self.predictions, label='Predicted')
        plt.plot(self.df["close"], label='Actual')
        plt.title(f"Predicted vs. Actual Closing Prices")
        plt.ylabel("Price, $USD")
        plt.legend()
        plt.show()
        
        
class ForecastPrice:
    '''Class is used to forecast Closing price of stock based on pre-trained LSTM model'''
    def __init__(self,
                 data : pd.DataFrame,
                 n_in : int = 100,
                 n_out : int = 14):
        
        self.df = data
        self.n_in = n_in
        self.n_out = n_out
        
        
        # Model class parameters
        self.model = None
        self.scaler = None
        self.close_scaler = None
        self.n_features = None
        self.forecasted_price = None
        
        
        
    ########################
    ### Helper Functions ###
    ########################
    def print_df(self):
        '''Prints DataFrame head'''
        print(self.df.head())
        
    def get_model_summary(self):
        '''Prints Model Summary'''
        try:
            self.model.summary()
        except:
            print("Model is not built.")
            
    def get_model(self):
        '''Returns Model Object'''
        return self.model
    
    def drop_columns(self, cols : list = ['open', 'high', 'low', 'volume']):
        '''Drops pd.DataFrame colums'''
        try:
            self.df.drop(columns=cols, inplace=True)
        except:
            print("Dataframe un-used columns alread dropped")
        
    def set_model_shape(self):
        '''Sets model shape by passing shape to n_features'''
        self.n_features = self.df.shape[1]
        
    def load_model(self, filename : str):
        '''Loads model'''
        self.model = load_model(filename)
        
    def forecast(self):
        '''Forecasts stock price based on pre-trained LSTM model'''
        
        # drop un-used columns from pd.DataFrame
        self.drop_columns()
        
        # set self.n_features parameter
        self.set_model_shape()
        
        # deep copy the pd.DataFrame containing technical indicators
        forecast_df = self.df.copy(deep=True)
        
        self.close_scaler = RobustScaler()
        self.close_scaler.fit(forecast_df[['close']])
        
        self.scaler = RobustScaler()
        transformed_forecast_df = pd.DataFrame(self.scaler.fit_transform(forecast_df), 
                                               columns=forecast_df.columns, 
                                               index=forecast_df.index).tail(self.n_in)
        
        
        # transform technical analysis data to np.array
        forecast_arr = np.array(transformed_forecast_df).reshape(1, 
                                                                 self.n_in, 
                                                                 self.n_features)
        
        # predicting off of the new data
        pred_y = self.model.predict(forecast_arr)
        
        # inverse_transform the predicted values back to original scale
        pred_y = self.close_scaler.inverse_transform(pred_y)[0]
        
        # parse perdicted values to pd.DataFrame, adjust date scale (index)
        preds = pd.DataFrame(pred_y, 
                     index=pd.date_range(start=forecast_df.index[-1]+timedelta(days=1), 
                                         periods=len(pred_y)), 
                     columns=[forecast_df.columns[0]])
        
        # set class variable
        self.forecasted_price = preds
        
        return preds
        
         
