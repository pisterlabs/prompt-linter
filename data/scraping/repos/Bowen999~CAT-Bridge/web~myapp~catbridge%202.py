"""
Package Name: CAT Bridge (Compounds And Trancrips Bridge)
Author: Bowen Yang
email: by8@ualberta
Homepage: 
Version: 0.5.1
Description: CAT Bridge (Compounds And Transcripts Bridge) is a robust tool built with the goal of uncovering biosynthetic mechanisms in multi-omics data, such as identifying genes potentially involved in compound synthesis by incorporating metabolomics and transcriptomics data. 

For more detailed information on specific functions or classes, use the help() function on them. For example:
help(your_package_name.your_function_name)


#  ██████╗ █████╗ ████████╗    ██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗
# ██╔════╝██╔══██╗╚══██╔══╝    ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝
# ██║     ███████║   ██║       ██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗  
# ██║     ██╔══██║   ██║       ██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝  
# ╚██████╗██║  ██║   ██║       ██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗
#  ╚═════╝╚═╝  ╚═╝   ╚═╝       ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝
"""

# Data
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import grangercausalitytests 
from causal_ccm.causal_ccm import ccm
import math
from sklearn.cross_decomposition import CCA
from fastdtw import fastdtw
from statsmodels.tsa.stattools import ccf
import skfuzzy as fuzz
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import linregress #linear regression



from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance



# Plot
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec
import seaborn as sns
from bioinfokit import analys, visuz
import networkx as nx
from PIL import Image
from adjustText import adjust_text
import textwrap


# Others
import shutil #move file
import getpass 
import os
import subprocess #run command line
import openai 
import datashader as ds 
from datashader import transfer_functions as tf
import getpass
from tqdm import tqdm



"""
***********  1. Pre-Processing ***********
#  ██████╗ █████╗ ████████╗    ██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗
# ██╔════╝██╔══██╗╚══██╔══╝    ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝
# ██║     ███████║   ██║       ██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗  
# ██║     ██╔══██║   ██║       ██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝  
# ╚██████╗██║  ██║   ██║       ██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗
#  ╚═════╝╚═╝  ╚═╝   ╚═╝       ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝
""" 

# *********** Read File ***********
def read_upload(file_name):
    """
    Read the uploaded file and return a dataframe
    supported file format: csv, txt, tsv, xls, xlsx
    """
    #if the last 3 letters of the file name is csv, we use csv module to read it
    if file_name[-3:] == 'csv':
        df = pd.read_csv(file_name, index_col=0)
    #if the last 3 letters of the file name is txt, we use csv module to read it
    elif file_name[-3:] == 'txt':
        df = pd.read_csv(file_name, sep='\t', index_col=0)
    #if the last 3 letters of the file name is tsv, we use csv module to read it
    elif file_name[-3:] == 'tsv':
        df = pd.read_csv(file_name, sep='\t', index_col=0)
    elif file_name[-3:] == 'xls':
        df = pd.read_excel(file_name, index_col=0)
    elif file_name[-4:] == 'xlsx':
        df = pd.read_excel(file_name, index_col=0)
    else:
        print('File format is not supported')
    return df


#get target compounds
def get_target(name, df):
    """
    find the target compound in the dataframe and return the row of the target
    """
    if name in df.index:
        name = df.loc[name].tolist()
    else:
        print('Target is not in the index of the dataframe')
    return name



#def normalize function, that allow log2, log10, and z-score
def normalize(df, method):
    """
    Normalize the data using the specified method.
    """
    if method == 'log2':
        df = np.log2(df + 1)
    elif method == 'log10':
        df = np.log10(df + 1)
    elif method == 'z-score':
        df = (df - df.mean()) / df.std()
    return df

  
    
#  ************* Scaling ***************
def scaling(df, method):
    """
    Scale the data using the specified method.
    """
    if method == 'min-max':
        scaler = MinMaxScaler()
        df = scaler.fit_transform(df)
    elif method == 'pareto':
        df = (df - df.mean()) / np.sqrt(df.std())


def scale_df(df):
    """Apply a log10 transformation and scale all values in a dataframe to the range 0-1."""
    
    # Apply a log10 transformation, add a small constant to avoid log(0)
    df = np.log10(df + 1e-10)
    
    # Check for NaNs and fill or remove them before scaling
    if df.isnull().values.any():
        df = df.fillna(method='ffill')  # you could also use 'bfill' or df.dropna()
        
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    return df_scaled


def scale_column(df, column_name):
    """
    Scale the values of a column using the MinMaxScaler.
    """
    # Create a scaler object
    scaler = MinMaxScaler()

    # Create a copy of the DataFrame to avoid modifying the original one
    df_scaled = df.copy()

    # Reshape the data to fit the scaler
    data = df[column_name].values.reshape(-1, 1)

    # Fit and transform the data
    df_scaled[column_name] = scaler.fit_transform(data)

    return df_scaled



# ************* Biological Replicates ***************
# ************** Max *******************************
def repeat_aggregation_max(df, design):
    """
    Aggregate biological replicates by taking the maximum value run for each row (gene/compound).
    
    Parameters:
        df (pandas DataFrame): The DataFrame to be aggregated.
        design (pandas DataFrame): The experimental design DataFrame.
    
    Returns:
        new_df (pandas DataFrame): The aggregated DataFrame.
    """
    # Calculate the number of suffixes
    num_suffixes = int(len(design.index) / len(design['group'].unique()))

    # Extract base column names and maintain their order
    base_cols = [col.rsplit('_', 1)[0] for col in df.columns if '_' in col]
    base_cols = sorted(set(base_cols), key=base_cols.index)

    # Create sub DataFrames and store their means
    mean_dfs = {}
    sub_dfs = {}
    for suffix in np.arange(1, num_suffixes + 1).astype(str):
        sub_df = df.filter(regex=f'_{suffix}$')
        sub_dfs[suffix] = sub_df
        mean_dfs[suffix] = sub_df.mean(axis=1)

    # Create a DataFrame to store which sub DataFrame has the highest mean for each row
    max_mean_df = pd.DataFrame(mean_dfs).idxmax(axis=1)

    # Create a new DataFrame, preserving the original index
    new_df = pd.DataFrame(index=df.index)

    # For each base column name
    for base_col in base_cols:
        for idx in new_df.index:
            # Determine which sub DataFrame to pull from for this row
            sub_df_idx = max_mean_df.loc[idx]

            # Get value from the appropriate sub DataFrame
            new_df.loc[idx, base_col] = sub_dfs[sub_df_idx].loc[idx, f"{base_col}_{sub_df_idx}"]


    new_df.columns = [str(col) + '_1' for col in new_df]
    mapping_dict = design['group'].to_dict()
    new_df.columns = new_df.columns.map(lambda x: mapping_dict[x] if x in mapping_dict else x)
    
    return new_df


# **************************** Mean **********************************
def repeat_aggregation_mean(df: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate biological replicates by taking the mean value run for each row (gene/compound).
    
    Parameters:
        df (pandas DataFrame): The DataFrame to be aggregated.
        design (pandas DataFrame): The experimental design DataFrame.
        
    Returns:
        new_df (pandas DataFrame): The aggregated DataFrame.
    """
    # Generate new column names from design DataFrame
    new_column_names = {i: design.loc[i]['group'] for i in df.columns}

    # Rename columns and average by new column names
    df_copy = df.rename(columns=new_column_names)
    df_copy = df_copy.T.groupby(level=0).mean().T
    
    return df_copy



# ************* Merge ***************
def merge_dataframes(dataframes):
    """
    Merge multiple dataframes based on the 'Name' column.

    Parameters:
        dataframes (list): A list of pandas DataFrames to be merged.

    Returns:
        merged_dataframe (pandas DataFrame): The merged DataFrame.
    """
    # Check if the input list is not empty
    if not dataframes:
        raise ValueError("The input list of dataframes is empty.")

    # Merge the dataframes one by one
    merged_dataframe = dataframes[0]
    for df in dataframes[1:]:
        merged_dataframe = merged_dataframe.merge(df, on='Name', how='outer')

    return merged_dataframe 



# ************ Find the increasing trend ***************
def calculate_trend_strength(df):
    """
    Function to calculate and sort the strength of the upward trend in each row of a DataFrame.

    This function calculates the trend strength (r-value) for each row in the DataFrame 
    using linear regression. It then adds a new column to the DataFrame with these r-values. 
    Finally, it sorts the DataFrame by these r-values in descending order and returns the 
    sorted DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be processed. 

    Returns:
    df_sorted_by_trend_strength (pandas.DataFrame): The input DataFrame, but with an added 
    column for trend strength and sorted by this column in descending order.
    """

    # Function to calculate the trend strength (r-value)
    def upward_trend_strength(row):
        _, _, rvalue, _, _ = linregress(x=range(len(row)), y=row)
        return rvalue

    # Apply function to each row to get trend strengths
    trend_strength = df.apply(upward_trend_strength, axis=1)

    # Add a new column to df with trend strengths
    df_with_trend_strength = df.assign(trend_strength=trend_strength)

    # Sort df by trend strength (r-value) in descending order
    df_sorted_by_trend_strength = df_with_trend_strength.sort_values(by='trend_strength', ascending=False)

    # Return sorted dataframe
    return df_sorted_by_trend_strength




# *************************** PCA (two dataframes) ***************************
def merge_and_reduce(df1, df2, n_components):
    """
    This function takes two dataframes with the same columns (sample names), performs Incremental PCA to reduce 
    the dimension of their index (IDs), and then merges them together. It ensures that both dataframes 
    have the same contribution to the final result by normalizing and reducing the dimension of each dataframe 
    separately before merging.

    Parameters:
    df1 (pandas.DataFrame): The first dataframe.
    df2 (pandas.DataFrame): The second dataframe.
    n_components (int): The number of dimensions for the output dataframe.

    Returns:
    pandas.DataFrame: The merged dataframe with reduced dimensions.
    """
    
    # Normalize each dataframe by their L2 norm
    scaler = StandardScaler()
    df1_norm = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns, index=df1.index)
    df2_norm = pd.DataFrame(scaler.fit_transform(df2), columns=df2.columns, index=df2.index)
    
    # Function to perform Incremental PCA and reduce dimensions
    def ipca_reduce(df, n_components):
        # Transpose the dataframe to reduce dimension of rows
        df_t = df.T

        # Apply Incremental PCA
        ipca = IncrementalPCA(n_components=n_components, batch_size=10)
        df_reduced = ipca.fit_transform(df_t)

        # Create a dataframe with the reduced dimension data
        df_reduced = pd.DataFrame(df_reduced, index=df_t.index, columns=[f'feature_{i}' for i in range(n_components)])

        # Transpose back to original form
        df_reduced_t = df_reduced.T

        return df_reduced_t
    
    # Reduce dimensions of each dataframe
    df1_reduced = ipca_reduce(df1_norm, n_components)
    df2_reduced = ipca_reduce(df2_norm, n_components)

    # Concatenate the two reduced dataframes
    df_merged = pd.concat([df1_reduced, df2_reduced])

    return df_merged







   
   
"""
************** 2 COMPUTE ******************
#  ██████╗ █████╗ ████████╗    ██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗
# ██╔════╝██╔══██╗╚══██╔══╝    ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝
# ██║     ███████║   ██║       ██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗  
# ██║     ██╔══██║   ██║       ██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝  
# ╚██████╗██║  ██║   ██║       ██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗
#  ╚═════╝╚═╝  ╚═╝   ╚═╝       ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝
""" 

# ************* 2.1 Granger **************************
def compute_granger(df, target, maxlag=1):
    """
    Compute the Granger causality score for each row in the dataframe.
    The X is each row in the dataframe and the Y is the target list.
    The returd value is the p-value of the F test for the maximum lag.
    
    Parameters:
        df (pandas DataFrame): The DataFrame to be aggregated.
        target (list): The target list.
    
    Returns:
        index_list (list): The list of row indices.
    
    """
    # Initialize lists to store results
    index_list = []
    p_value_list = []

    # Loop through each row in the dataframe
    for idx, row in df.iterrows():
        # Skip if the row or the target list has constant values
        if np.std(row.values) == 0 or np.std(target) == 0:
            continue

        # Combine the row and target into a DataFrame
        data = pd.concat([pd.Series(target), pd.Series(row.values)], axis=1)

        # Perform the Granger causality test
        try:
            result = grangercausalitytests(data, maxlag=maxlag, verbose=False)

            # Extract the p-value of the F test for the maximum lag
            p_value = result[maxlag][0]['ssr_ftest'][1]
            # p_value = 1-p_value

            # Append results to the lists
            index_list.append(idx)
            p_value_list.append(p_value)
        except Exception as e:
            #add NA if there is an error
            index_list.append(idx)
            p_value_list.append(np.nan)
            
            # print(f"Error at index {idx}: {str(e)}")

    # Create a new dataframe to store results
    results_df = pd.DataFrame({
        'Name': index_list,
        'Granger': p_value_list
    })

    return results_df


def compute_reverse_granger(df, target, maxlag=1):
    """
    X is target and Y is each row in the dataframe (df).
    """
    
     # Initialize lists to store results
    index_list = []
    p_value_list = []

    # Loop through each row in the dataframe
    for idx, row in df.iterrows():
        # Skip if the row or the target list has constant values
        if np.std(row.values) == 0 or np.std(target) == 0:
            continue

        # Combine the row and target into a DataFrame
        data = pd.concat([pd.Series(row.values), pd.Series(target)], axis=1)

        # Perform the Granger causality test
        try:
            result = grangercausalitytests(data, maxlag=maxlag, verbose=False)

            # Extract the p-value of the F test for the maximum lag
            p_value = result[maxlag][0]['ssr_ftest'][1]
            p_value = 1-p_value

            # Append results to the lists
            index_list.append(idx)
            p_value_list.append(p_value)
        except Exception as e:
            #add NA if there is an error
            index_list.append(idx)
            p_value_list.append(np.nan)
            
            # print(f"Error at index {idx}: {str(e)}")

    # Create a new dataframe to store results
    results_df = pd.DataFrame({
        'Name': index_list,
        'Granger': p_value_list
    })

    return results_df



def granger_list(A, B, maxlag):
    """
    Compute granger test P value for listA (X), and listB (y) 
    """
    
    # Combine the two lists into a 2D array
    data = np.column_stack((A, B))
    
    # Perform the Granger causality test and return the result
    result = grangercausalitytests(data, maxlag, verbose=False)
    #result = grangercausalitytests(data, maxlag)

    # In this function, we'll return just the p-value for each lag.
    # You may want to modify this to return other information.
    p_values = {lag: result[lag][0]['ssr_ftest'][1] for lag in result.keys()}
    
    return p_values







# ****************** 2.2 CCM (Convergent cross mapping)*******************************
def compute_ccm(df, target, E=3, tau=1):
    # Placeholder for result list
    results = [] 

    for name, data in df.iterrows():
        # Get data from the row (as the name is now the index)
        data_values = data.values
        
        if np.any(np.isnan(data_values)) or np.any(np.isnan(target)):
            print("data_values:", data_values)
            print("target:", target)
            continue  # Skip the current iteration if NaN is found


        # Calculate ccm
        ccm1 = ccm(data_values, target, tau, E)
        
        # Get causality value
        causality_val = ccm1.causality()[0]
        
        # Append to results list
        results.append([name, causality_val])

    # Convert results list to DataFrame
    result_df = pd.DataFrame(results, columns=['Name', 'CCM'])
    
    # Sort by Causality in descending order and reset the index
    result_df = result_df.sort_values(by='CCM', ascending=False).reset_index(drop=True)
    
    return result_df







# ******************** 2.3 CCA (Canonical correlation) ********************
def compute_cca(df, target, n_components=1):
    # Ensure the target data is 2D (samples x features)
    metabolite_concentration = np.array(target).reshape(-1, 1)
    results_dict = {"Name": [], "CCA": []}  # Initialize a dictionary to store the results

    for index, row in df.iterrows():
        # Reshape the gene_expression data to be 2D (samples x features)
        gene_expression = np.array(row).reshape(-1, 1)

        # It's essential to have more than one sample to perform CCA,
        # so if there's only one sample, append NaN and continue
        if gene_expression.shape[0] <= 1 or metabolite_concentration.shape[0] <= 1:
            results_dict["Name"].append(index)
            results_dict["CCA"].append(np.nan)
            continue

        try:
            # Instantiate CCA
            cca = CCA(n_components=n_components)
            cca.fit(gene_expression, metabolite_concentration)

            # Transform the data based on the canonical correlation vectors
            gene_c, metabolite_c = cca.transform(gene_expression, metabolite_concentration)

            # Compute the correlation between the transformed data
            correlation = np.corrcoef(gene_c.T, metabolite_c.T)[0, 1]
            results_dict["Name"].append(index)
            results_dict["CCA"].append(correlation)

        except ValueError as e:
            results_dict["Name"].append(index)
            results_dict["CCA"].append(np.nan)  # Append NaN for rows causing errors

    return pd.DataFrame(results_dict)  # Convert the results dictionary to a DataFrame and return it





# ******************* 2.4 DTW (Dynamic Time Warping) *************************
def compute_dtw(df, target):
    results_dict = {"Name": [], "DTW": []}  # Initialize a dictionary to store the results
    
    # Prepare the MinMax scaler
    scaler = MinMaxScaler()
    
    for index, row in df.iterrows():
        # Convert the row data and target to numpy arrays
        array1, array2 = np.array(row).reshape(-1, 1), np.array(target).reshape(-1, 1)
        
        # Normalize the arrays
        array1_norm = scaler.fit_transform(array1)
        array2_norm = scaler.fit_transform(array2)
        
        # Compute the DTW distance using the fastdtw function
        distance, path = fastdtw(array1_norm, array2_norm)
        
        # Append the index and DTW distance to the results dictionary
        results_dict["Name"].append(index)
        results_dict["DTW"].append(distance)
    
    return pd.DataFrame(results_dict)  # Convert the results dictionary to a DataFrame and return it






# ************************* 2.5 CCF *****************************
def compute_ccf(df, target, lag=1):
    results_dict = {"Name": [], "CCF": []}  # Initialize a dictionary to store the results

    for index, row in df.iterrows():
        gene_expression = np.array(row)

        # Ensure gene_expression and target have equal lengths, or handle it accordingly
        # One simple method could be truncating the longer array, but other methods might be more suitable depending on the context
        min_length = min(len(gene_expression), len(target))
        gene_expression = gene_expression[:min_length]
        truncated_target = np.array(target)[:min_length]

        # Compute the cross-correlation function
        cross_correlation = ccf(gene_expression, truncated_target)

        # Extract the cross-correlation at the specified lag
        # Handling the case where the specified lag is out of bounds by appending NaN
        if abs(lag) >= len(cross_correlation):
            results_dict["Name"].append(index)
            results_dict["CCF"].append(np.nan)
        else:
            lag_value = cross_correlation[lag] if lag >= 0 else cross_correlation[lag - 1]
            results_dict["Name"].append(index)
            results_dict["CCF"].append(lag_value)

    return pd.DataFrame(results_dict)  # Convert the results dictionary to a DataFrame and return it








# ***************** 2.6 Spearman *********************
def compute_spearman(df, target):
    # Initialize lists to store results
    index_list = []
    corr_list = []
    p_value_list = []

    # Loop through each row in the dataframe
    for idx, row in df.iterrows():
        # Compute Spearman correlation and p-value
        corr, p_value = spearmanr(row, target)

        # Append results to the lists
        index_list.append(idx)
        corr_list.append(corr)
        p_value_list.append(p_value)

    # Create a new dataframe to store results
    results_df = pd.DataFrame({
        'Name': index_list,
        'Spearman': corr_list
        #'P-value': p_value_list
    })

    return results_df



# ***************** 2.7 Pearson *********************
def compute_pearson(df, target):
    # Initialize lists to store results
    index_list = []
    corr_list = []
    p_value_list = []

    # Loop through each row in the dataframe
    for idx, row in df.iterrows():
        # Compute Pearson correlation and p-value
        corr, p_value = pearsonr(row, target)

        # Append results to the lists
        index_list.append(idx)
        corr_list.append(corr)
        p_value_list.append(p_value)

    # Create a new dataframe to store results
    results_df = pd.DataFrame({
        'Name': index_list,
        'Pearson': corr_list
        #'P-value': p_value_list
    })

    return results_df







# ************** fuzz c mean clustering *********************
def clustering(df, n_clusters):
    # Prepare the data
    data = df.values.T  # The cmeans function expects data to be of shape (n_features, n_samples)
    
    # Perform fuzzy c-means clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, n_clusters, 2, error=0.005, maxiter=1000)
    
    # Get the cluster assignment for each sample
    # The argmax function is used to assign each sample to the cluster for which it has the highest membership value
    cluster_assignment = u.argmax(axis=0)
    
    # Create a DataFrame to hold the results
    results_df = pd.DataFrame({'Name': df.index, 'Cluster': cluster_assignment})
    
    return results_df





# ts Clustering
def ts_clustering(df, n_clusters):
    """
    Cluster the time series data in the input dataframe using the TS-KMeans algorithm.
    
    Parameters:
        df (pandas.DataFrame): The input dataframe.
        n_clusters (int): The number of clusters to create.
    """
    # Convert DataFrame to NumPy array for compatibility with tslearn
    data = df.values

    # Rescale the time series data so that their mean is 0 and their standard deviation is 1
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
    data_scaled = scaler.fit_transform(data)

    # Create the KMeans model
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", max_iter=5, random_state=0)

    # Fit the model to the data
    km.fit(data_scaled)

    # Get the cluster labels for each time series
    labels = km.labels_

    # Add the labels as a new column in the original DataFrame
    df['Cluster'] = labels
    df = df['Cluster']

    return df




# ********* Noontide ************
def find_noontide(df, row_name):
    """
    Find the column with the highest value in the row, and return the column, and the column after it (name).
    If the column with the highest value is the last column, then return the column with the second highest value, and the column after it (name).
    
    Parameters:
        df (pandas DataFrame): The DataFrame has been aggregated.
        row_name (str): The name of the row to be detected.
        
    Returns:
        column_n (str): The name of the column with the highest value.
    """
    # Get the row with the specified name
    row = df.loc[row_name].copy()

    # Convert to numeric and handle errors by replacing non-numeric values with NaN
    row = pd.to_numeric(row, errors='coerce')

    # Handle rows that are all NaN
    if row.isna().all():
        raise Exception(f"The row '{row_name}' contains only non-numeric or missing values.")

    # Identify the column with the highest value
    column_n = row.idxmax()
    col_idx = list(df.columns).index(column_n)

    # If column with max value is last column, then find the second highest column
    if col_idx == len(df.columns) - 1:
        row[column_n] = np.nan # set value in column_n to NaN
        column_n = row.idxmax() # get column with max value now

    # Find column after column_n
    col_idx = list(df.columns).index(column_n)
    if col_idx < len(df.columns) - 1: # Make sure it's not the last column
        column_n_plus_1 = df.columns[col_idx + 1]
    else:
        raise Exception("There is no column after the column with the maximum value.")

    # Keep only column n and column n+1
    df_filtered = df[[column_n, column_n_plus_1]]

    return df_filtered.columns



# ********* df for fc ************
def df_for_fc(df1, target, df2, design):
    """
    Gnerate the design matrix and matrix for computing the FC score.
    
    Parameters:
        df1 (pandas DataFrame): The DataFrame has been aggregated (processed_metabo).
        target (str): The name of the row to be detected (Capsaicin).
        df2 (pandas DataFrame): The DataFrame for fc computing (gene).
        design (pandas DataFrame): study design, the samle and group information.
    """
    noontide = find_noontide(df1, target)
    design_fc = design[design['group'].isin(noontide)]
    matrix_fc = df2[design_fc.index]

    # Saving to CSV files instead of returning
    design_fc.to_csv('result/design_fc.csv')
    matrix_fc.to_csv('result/matrix_fc.csv')
    
    


def no_repeat_fc(df, noontide):
    """
    Compute the FC score for each metabolite. This function is used for Study Design that has no biological repeats.
    
    Parameters:
        df (pandas DataFrame): The DataFrame for fc computing (gene).
        noontide (list): The list of the column names of the noontide samples. (can be obtained from find_noontide function)
    
    Returns:
        new_df (pandas DataFrame): A new DataFrame with the FC score.
    """
    
    # Create a new DataFrame as a copy of the original
    new_df = df.copy()
    
    # Compute log2FoldChange
    new_df['log2FoldChange'] = new_df[noontide[0]] / new_df[noontide[1]]
    new_df['log2FoldChange'] = new_df['log2FoldChange'].apply(lambda x: math.log2(x if x > 0 else np.finfo(float).eps))
    
    # Replace infinities with 0
    new_df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Replace NaNs with 0
    new_df['log2FoldChange'].fillna(0, inplace=True)
    
    # Make the range of log2FoldChange from 0-1
    scaler = MinMaxScaler()
    new_df['log2FoldChange'] = scaler.fit_transform(new_df[['log2FoldChange']])
    
    # Keep only the 'log2FoldChange' column in the new DataFrame
    new_df = new_df[['log2FoldChange']]
    
    return new_df



# ********* FC Compute ********
def fc_comp():
    """
    Compute the fold change using R script(FC.R)
    """
    result = subprocess.run(['Rscript', 'FC.R'], stdout=subprocess.PIPE)
    fc = read_upload('result/fc.csv')
    fc.index.name = 'Name'
    #for value in fc['log2FoldChange'], do scaling to makeit range from 0-1
    scaler = MinMaxScaler()
    # Apply the scaler to the 'log2FoldChange' column 
    fc['log2FoldChange'] = -1 * fc['log2FoldChange']
    # fc['log2FoldChange'] = scaler.fit_transform(fc[['log2FoldChange']])
    return fc



# ***************  compute score ******************
def score(df):
    """
    Compute a score based on specified columns of a dataframe, and rank the rows based on the score.
    """
    # Exclude the 'Name' column and compute row sums
    df['Score'] = df.drop(columns=['Name']).sum(axis=1)
    
    # Rank rows from high to low based on the sum, '1' being the highest rank
    df['Rank'] = df['Score'].rank(method='min', ascending=False)
    
    # Sort dataframe by rank
    df_sorted = df.sort_values('Rank')
    # df_sorted.set_index('Rank', inplace=True)
    # df_sorted = df_sorted[['Name', 'Score', 'Rank']]
    
    return df_sorted


def compute_score(df):
    scaler = MinMaxScaler()
    granger_scaled = scaler.fit_transform(df[['Granger']])
    log2fc_scaled = scaler.fit_transform(df[['log2FoldChange']])
    
    if 'Description_Score' in df.columns:
        df['Score'] = np.sqrt((granger_scaled - log2fc_scaled)**2 + df['Description_Score']**2)
    else:
        df['Score'] = np.sqrt((granger_scaled - log2fc_scaled)**2)
    
    return df



# Annotation -> Score
keywords_scores = {'ase': 0.2, 'enzyme': 0.2, 'synthase': 0.2}
def add_annotation_score(df, keywords=keywords_scores):
    """
    This function adds a new column 'Description_Score' to the input dataframe. The values in this column are computed
    based on the 'Description' column and the keywords dictionary.
    """
    def compute_description_score(description):
        try:
            if pd.isna(description):  # Check if the value is NaN
                return 0.1
            words = description.split()  # Split the description into words
            max_score = None  # Initialize max score
            for word in words: 
                for keyword, score in keywords.items(): 
                    if word.endswith(keyword):  # If the word ends with a keyword
                        if max_score is None or score > max_score:  # If score is higher than current max score
                            max_score = score  # Update max score
            if max_score is not None:  # If a keyword was found
                return max_score  # Return max score
            return 0  # Return 0 if no keywords are found
        except Exception as e:
            print(f"An error occurred: {e}")
            return 0  # Return 0 in case of an error

    df['Description_Score'] = df['Description'].apply(compute_description_score)
    return df





# ************* Upward trend strength *************
def trend_strength(df):
    """
    Function to calculate and sort the strength of the upward trend in each row of a DataFrame.

    This function calculates the trend strength (r-value) for each row in the DataFrame 
    using linear regression. It then adds a new column to the DataFrame with these r-values. 
    Finally, it sorts the DataFrame by these r-values in descending order and returns the 
    sorted DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be processed. 

    Returns:
    df_sorted_by_trend_strength (pandas.DataFrame): The input DataFrame, but with an added 
    column for trend strength and sorted by this column in descending order.
    """

    # Function to calculate the trend strength (r-value)
    def upward_trend_strength(row):
        _, _, rvalue, _, _ = linregress(x=range(len(row)), y=row)
        return rvalue

    # Apply function to each row to get trend strengths
    trend_strength = df.apply(upward_trend_strength, axis=1)

    # Add a new column to df with trend strengths
    df_with_trend_strength = df.assign(trend_strength=trend_strength)

    # Sort df by trend strength (r-value) in descending order
    df_sorted_by_trend_strength = df_with_trend_strength.sort_values(by='trend_strength', ascending=False)

    # Return sorted dataframe
    return df_sorted_by_trend_strength



# ****************** Importance *******************
def top_important_features(df, n_top):
    """
    This function calculates the importance of features based on a PCA model. The importance is calculated 
    as the sum of the squares of the PCA loadings for each feature. The function then returns the top n 
    features based on their importance.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame where each row corresponds to a sample and each column corresponds 
                       to a feature (e.g., gene expression values for different genes). The index of the DataFrame
                       is assumed to represent sample identifiers and the columns of the DataFrame are assumed to 
                       represent feature names.
    n_top (int): The number of top features to return based on their calculated importance.

    Returns:
    pd.DataFrame: A DataFrame containing the top n features and their corresponding importances, sorted by importance.
    """

    try:
        # Transpose your data so that rows are time points and columns are genes
        df_transposed = df.transpose()

        # Initialize PCA object
        pca = PCA()

        # Fit the model
        pca.fit(df_transposed)

        # Get the loadings for each gene
        loadings = pca.components_

        # Calculate the importance of each gene as the sum of squares of its loadings
        importance = np.sum(loadings**2, axis=0)

        # Create a DataFrame of importances and gene names
        importance_df = pd.DataFrame({'Gene': df_transposed.columns, 'Importance': importance})

        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)

        # Return the top n genes
        return importance_df.head(n_top)

    except Exception as e:
        print("An error occurred while computing the top important features.")
        print(str(e))
        return None






# ****************** Compute Score ***********************
def compute_modulus(a, b, c=None):
    if c is None:
        return (a ** 2 + b ** 2) ** 0.5
    else:
        return (a ** 2 + b ** 2 + c ** 2) ** 0.5

def compute_score(df, col):
    col = str(col)
    if col == 'Granger':
        df[col] = 1 - df[col]
    # Initializing MinMaxScaler
    scaler = MinMaxScaler()
    
    # Fitting and transforming the specified columns
    df[['norm_col', 'norm_fc']] = scaler.fit_transform(df[[col, 'log2FoldChange']])
    
    # If 'Description_Score' column exists in the DataFrame
    if 'Description_Score' in df.columns:
        df['Score'] = df.apply(lambda row: compute_modulus(row['norm_col'], row['norm_fc'], row['Description_Score']), axis=1)
    else:
        df['Score'] = df.apply(lambda row: compute_modulus(row['norm_col'], row['norm_fc']), axis=1)
    
    df.sort_values(by='Score', ascending=False, inplace=True)
    
    
    df['Score'] = MinMaxScaler().fit_transform(df[['Score']])
    df = df[['Name', col, 'log2FoldChange', 'Cluster', 'Score']]
    df.reset_index(drop=True, inplace=True)
    df.index = df.index + 1
    
    return df















"""
#********* 3 PLOT FUNCTION ********   
#  ██████╗ █████╗ ████████╗    ██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗
# ██╔════╝██╔══██╗╚══██╔══╝    ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝
# ██║     ███████║   ██║       ██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗  
# ██║     ██╔══██║   ██║       ██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝  
# ╚██████╗██║  ██║   ██║       ██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗
#  ╚═════╝╚═╝  ╚═╝   ╚═╝       ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝
"""

    
# Line Plot
def plot_line(df, target, save_path=None):
    """
    Plots a line graph of the target row from the dataframe.
    Args:
    df (pd.DataFrame): DataFrame to plot.
    target (str): Target row to plot.
    """
    try:
        # Access the target row
        target_row = df.loc[target]

        # Create the line plot
        plt.figure(figsize=(12,3))
        plt.plot(target_row.index, target_row.values)
        
        # Add labels and title
        plt.title(f'{target}')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


# Heatamp

# def plot_heatmap(dataframe, palette='vlag', figsize=(6, 8), row_threshold=50, n_clusters=1000, save_path=None):
#     """
#     Plot the heatmap of the dataframe. If the number of rows is greater than row_threshold, row labels are not shown.
#     Args:
#     dataframe (pd.DataFrame): DataFrame to plot.
#     palette (str, optional): Palette to use for the plot. Defaults to 'vlag'.
#     figsize (tuple, optional): Size of the plot. Defaults to (10, 8).
#     row_threshold (int, optional): Threshold for number of rows to determine whether to show row labels. Defaults to 50.
#     n_clusters (int, optional): Number of clusters to use if rows are over row_threshold. Defaults to 10.
#     save_path (str, optional): If provided, save the plot to this path. Otherwise, the plot is shown using plt.show(). Defaults to None.
#     """
#     try:
#         # Round the dataframe to the nearest integer
#         dataframe = dataframe.round(0).astype(int)

#         # If the number of rows is greater than row_threshold, perform clustering and replace rows with cluster centroids
#         if dataframe.shape[0] > row_threshold:
#             kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(dataframe)
#             dataframe['cluster'] = kmeans.labels_
#             dataframe = dataframe.groupby('cluster').mean()

#         # Remove the rows that have all 0 values
#         dataframe = dataframe.loc[~(dataframe==0).all(axis=1)]

#         # Generate the heatmap
#         row_labels = dataframe.shape[0] <= row_threshold
#         if row_labels:
#             plt.figure(figsize=figsize)
#             sns.heatmap(dataframe, cmap=palette, yticklabels=row_labels, xticklabels=True)
#         else:
#             sns.clustermap(dataframe, cmap=palette, yticklabels=row_labels, xticklabels=True, figsize=figsize, col_cluster=False)

#         # Set x axis label
#         plt.xlabel(dataframe.columns.name)

#         # If row_labels is False, add a y-axis label 'features'
#         if not row_labels:
#             plt.ylabel('features')

#         if save_path:
#             plt.savefig(save_path)
#         else:
#             plt.show()

#     except Exception as e:
#         print(f"An error occurred: {e}")
def plot_heatmap(dataframe, palette='vlag', figsize=(6, 8), row_threshold=50, n_clusters=None, save_path=None):
    """
    Plot the heatmap of the dataframe. If the number of rows is greater than row_threshold, row labels are not shown.
    Args:
    dataframe (pd.DataFrame): DataFrame to plot.
    palette (str, optional): Palette to use for the plot. Defaults to 'vlag'.
    figsize (tuple, optional): Size of the plot. Defaults to (6, 8).
    row_threshold (int, optional): Threshold for number of rows to determine whether to show row labels. Defaults to 50.
    n_clusters (int, optional): Number of clusters to use if rows are over row_threshold. Defaults to None.
    save_path (str, optional): If provided, save the plot to this path. Otherwise, the plot is shown using plt.show(). Defaults to None.
    """
    try:
        # Round the dataframe to the nearest thousandth
        dataframe = dataframe.round(2)

        # If the number of rows is greater than row_threshold and n_clusters is not None, perform clustering
        if n_clusters is not None and dataframe.shape[0] > row_threshold:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(dataframe)
            dataframe['cluster'] = kmeans.labels_
            dataframe = dataframe.groupby('cluster').mean()

        # Remove the rows that have all 0 values
        dataframe = dataframe.loc[~(dataframe==0).all(axis=1)]

        # Generate the heatmap
        row_labels = dataframe.shape[0] <= row_threshold
        if row_labels:
            plt.figure(figsize=figsize)
            heatmap = sns.heatmap(dataframe, cmap=palette, yticklabels=row_labels, xticklabels=True)
            plt.ylabel('')  # Remove y-axis label
        else:
            g = sns.clustermap(dataframe, cmap=palette, yticklabels=row_labels, xticklabels=True, figsize=figsize, col_cluster=False)
            g.ax_heatmap.set_ylabel('')  # Remove y-axis label
            plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), text='')  # Remove y-axis labels

        # Set x axis label
        plt.xlabel(dataframe.columns.name)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

# ********************* PCA ****************************
def plot_pca(gene, design, n_clusters, save_path=None):
    """
    Perform PCA on the gene expression data, and plot the result.
    
    Parameters:
        gene: the gene expression dataframe
        design: the design dataframe
        n_clusters: the number of clusters to use in K-means clustering
    """
    try:
        # Transpose your DataFrame, as PCA works on the features (columns), not on the samples (rows)
        gene_transposed = gene.T

        # Perform PCA on your data
        pca = PCA(n_components=2)  # here we ask for the first two principal components
        pca_result = pca.fit_transform(gene_transposed)

        # Convert the PCA result to a DataFrame
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

        # Add a 'sample' column
        pca_df['sample'] = gene_transposed.index

        # Get group information from the design dataframe
        pca_df = pca_df.merge(design[['group']], left_on='sample', right_index=True)

        fig, ax = plt.subplots(figsize=(6, 8))

        if n_clusters > 0:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pca_df[['PC1', 'PC2']])
            pca_df['Cluster'] = kmeans.labels_

            # Plot the points on the scatterplot
            scatter = sns.scatterplot(x="PC1", y="PC2", hue="group", data=pca_df, palette="Paired", s=100, alpha=0.7, ax=ax)

            # For each cluster, add a circle at the mean coordinates with radius proportional to the standard deviation
            for cluster in set(kmeans.labels_):
                cluster_points = pca_df[pca_df['Cluster'] == cluster][['PC1', 'PC2']]
                # Calculate mean and standard deviation for the cluster
                cluster_mean = cluster_points.mean().values
                cluster_std = cluster_points.std().values
                # Add a circle at the mean coordinates with radius=stddev
                circle = Circle(cluster_mean, np.linalg.norm(cluster_std), alpha=0.1)
                ax.add_artist(circle)
        else:
            # Plot the points on the scatterplot without clustering
            scatter = sns.scatterplot(x="PC1", y="PC2", hue="group", data=pca_df, palette="Paired", s=100, alpha=0.7, ax=ax)

        # Check if the legend exists before trying to remove it
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        # Annotate points on the graph with the sample names
        texts = []
        for i, sample in enumerate(pca_df['sample']):
            texts.append(plt.text(pca_df.iloc[i].PC1, pca_df.iloc[i].PC2, sample, color='gray'))
            
        # Adjust the labels to avoid overlap
        adjust_text(texts)

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        # plt.title('PCA', fontweight='bold')

        # Hide X and Y values
        plt.xticks([])
        plt.yticks([])
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")



# ****************** Top Important Features ******************
def plot_top_features(df, color='crest', save_path=None):
    """
    This function creates a scatter plot of the importances of the top n features, as returned by the 
    top_important_features function. The color palette is applied based on bins of 'Importance'.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame as returned by the top_important_features function.
    color (str): The color map to use for the scatter plot.

    Returns:
    None
    """

    try:
        # Bin the 'Importance' into categories
        df['Importance_Category'] = pd.qcut(df['Importance'], q=4)

        # Create the scatter plot
        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = sns.scatterplot(data=df, y='Gene', x='Importance', hue='Importance_Category', size='Importance', palette=color, sizes=(50, 200), ax=ax)
        
        # Set title and labels
        ax.set_xlabel('Importance')
        ax.set_ylabel(' ')

        # Split long y labels into two lines
        y_labels = [textwrap.fill(label.get_text(), width=20) for label in scatter.get_yticklabels()]
        ax.set_yticklabels(y_labels)

        # Remove legend
        scatter.legend_.remove()

        # Adjust the space for y label
        plt.subplots_adjust(left=0.3)
          
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        return plt

    except Exception as e:
        print("An error occurred while creating the scatter plot.")
        print(str(e))




# Volcano Plot
# def plot_volcano(path, lfc_threshold, padj_threshold):
#     """
#     path: the path to the csv file
#     lfc_threshold: the log2 fold change threshold
#     padj_threshold: the adjusted p-value threshold
#     """
#     # Import and preprocess data
#     fc_for_volcano = pd.read_csv(path)
#     fc_for_volcano.reset_index(inplace=True)
#     fc_for_volcano.rename(columns={'index':'Name'}, inplace=True)
#     gene_exp = fc_for_volcano
#     gene_exp = gene_exp.dropna(subset=['padj'])

#     # Check if at least 10 gene names exist
#     genenames = gene_exp['Name'].head(10) if len(gene_exp['Name']) >= 10 else None

#     # Create plot
#     plt.rcParams['figure.figsize'] = [6, 6]
#     visuz.GeneExpression.volcano(df=gene_exp, 
#                                 lfc='log2FoldChange', pv='padj', sign_line=True,
#                                 lfc_thr=(lfc_threshold, lfc_threshold), pv_thr=(padj_threshold, padj_threshold),
#                                 plotlegend=True, legendpos='upper right', legendanchor=(1.46,1),
#                                 color=('maroon','gainsboro','steelblue'), theme='whitesmoke',
#                                 valpha=1, dotsize=5,
#                                 geneid = 'Name'
#                                 # genenames = tuple(genenames) if genenames is not None else None,
#                                 )

#     # plt.savefig('result/volcano_plot.png')
#     img = Image.open('volcano.png')  # replace with your image file path if not in the same directory
    
#     shutil.move("volcano.png", "result/plot/volcano.png")
#     if os.path.exists("volcano.png"):
#         os.remove("volcano.png")
    
#     # Create a figure and a set of subplots with specified size
#     fig, ax = plt.subplots()
#     # Display the image
#     ax.imshow(img)
#     # Remove the axis
#     ax.axis('off')
#     # Show the figure
#     plt.show()
def plot_volcano(path, lfc_threshold, padj_threshold):
    """
    path: the path to the csv file
    lfc_threshold: the log2 fold change threshold
    padj_threshold: the adjusted p-value threshold
    """
    # Import and preprocess data
    fc_for_volcano = pd.read_csv(path)
    fc_for_volcano.reset_index(inplace=True)
    fc_for_volcano.rename(columns={'index':'Name'}, inplace=True)
    gene_exp = fc_for_volcano
    gene_exp = gene_exp.dropna(subset=['padj'])

    # Check if at least 10 gene names exist
    genenames = gene_exp['Name'].head(10) if len(gene_exp['Name']) >= 10 else None

    # Create plot
    plt.rcParams['figure.figsize'] = [6, 6]
    visuz.GeneExpression.volcano(df=gene_exp, 
                                lfc='log2FoldChange', pv='padj', sign_line=True,
                                lfc_thr=(lfc_threshold, lfc_threshold), pv_thr=(padj_threshold, padj_threshold),
                                plotlegend=True, legendpos='upper right', legendanchor=(1.46,1),
                                color=('maroon','gainsboro','steelblue'), theme='whitesmoke',
                                valpha=1, dotsize=5,
                                geneid = 'Name'
                                # genenames = tuple(genenames) if genenames is not None else None,
                                )

    # plt.savefig('result/volcano_plot.png')
    img = Image.open('volcano.png')  # replace with your image file path if not in the same directory
    
    # shutil.move("volcano.png", "result/plot/volcano.png")
    # if os.path.exists("volcano.png"):
    #     os.remove("volcano.png")
    
    # Create a figure and a set of subplots with specified size
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(img)
    # Remove the axis
    ax.axis('off')
    # Show the figure
    plt.show()






# ********************** Time Series Clusters **********************
# Plot each line, and put them in the same figure (for each Cluster)
def plot_ts_clusters(result, processed_gene, palette_name='crest', save_fig=False):
    result = result[['Name', 'Cluster']]
    data = merge_dataframes([result, processed_gene])

    # Drop the 'Name' column
    data = data.drop(columns=['Name'])

    # Reset index, group by 'Cluster', and set the index back
    data = data.reset_index()
    grouped = data.groupby('Cluster')

    # Set color palette
    palette = sns.color_palette(palette_name, 12)  # The palette_name palette has maximum 12 distinct colors

    # Iterate over groups (clusters) and plot each one
    for name, group in grouped:
        group = group.set_index('index')  # set the index back to 'index'
        group = group.drop(columns='Cluster')  # drop the 'Cluster' column

        plt.figure(figsize=(10, 3))
        for i, feature in enumerate(group.index):
            plt.plot(group.columns, group.loc[feature], color=palette[i % len(palette)])

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Cluster: {name}', weight='bold')
        # plt.savefig('result/plot/ts_cluster.png')
        # plt.show()

        if save_fig:
            # Save the figure with a unique filename for each cluster
            plt.savefig(f"result/plot/ts_cluster_{name}.png")
        else:
            plt.show()


def plot_ts_clusters_average(result, processed_gene, palette_name='Paired', save_fig=None):
    """
    Plot each line, and put them in the same figure (for each Cluster)
    
    Parameters:
        result: the result dataframe including the 'Name' and 'Cluster' columns
        processed_gene: the dataframe (with the 'Name' column) to plot
    
    result: the result dataframe from the clustering algorithm
    """
    data = merge_dataframes([result, processed_gene])

    # Drop the 'Name' column
    data = data.drop(columns=['Name'])

    # Reset index, group by 'Cluster', and set the index back
    data = data.reset_index()
    grouped = data.groupby('Cluster')

    # Set color palette
    palette = sns.color_palette(palette_name, 12)  # The palette_name palette has maximum 12 distinct colors

    # Iterate over groups (clusters) and plot each one
    for name, group in grouped:
        group = group.set_index('index')  # set the index back to 'index'
        group = group.drop(columns='Cluster')  # drop the 'Cluster' column

        # Convert the group dataframe to long form
        long_form = pd.melt(group.reset_index(), id_vars='index')

        plt.figure(figsize=(20, 8))
        
        # Use seaborn lineplot to plot each line (gene) and the average line with 95% CI
        sns.lineplot(x='variable', y='value', data=long_form, color=palette[int(name) % len(palette)], ci=95)

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'Cluster: {name}', weight='bold')
        
        if save_fig:
            # Save the figure with a unique filename for each cluster
            plt.savefig(f"result/plot/ts_cluster_{name}.png")
        else:
            plt.show()




# ******************* Network ***********************
# def plot_network(data, target_index, num_nodes=20, save_path=None):
#     """
#     Plot the network graph of the top num_nodes similar nodes for the target.
    
#     Parameters:
#         data: the dataframe containing the similarity data
#         target_index: the index of the target node
#         num_nodes: the number of nodes to plot
#     """
#     # Compute the similarity
#     similarity = cosine_similarity(data)
#     similarity_df = pd.DataFrame(similarity, index=data.index, columns=data.index)
    
#     # Get the top num_nodes similar nodes for the target
#     target_similarities = similarity_df.loc[target_index].sort_values(ascending=False)[1:num_nodes+1].to_dict()

#     # Create a network graph
#     G = nx.Graph()

#     # Add nodes and edges
#     for node, similarity in target_similarities.items():
#         G.add_edge(target_index, node, weight=similarity)
    
#     # Draw the network
#     plt.figure(figsize=(10,8))
#     pos = nx.spring_layout(G)
#     colors = ['red' if node == target_index else 'moccasin' for node in G.nodes()]
    
#     nx.draw_networkx_nodes(G, pos, node_color=colors)
#     nx.draw_networkx_labels(G, pos, font_size=8)
#     nx.draw_networkx_edges(G, pos, edge_color='gray', width=[G[u][v]['weight'] for u,v in G.edges()], alpha=0.7)

#     # plt.title(f'Top {num_nodes} similar features to the target', fontsize=10)
#     plt.axis('off')  # to turn off the frame
    
#     if save_path:
#         plt.savefig(save_path)
        
#     else:
#         plt.show()

def plot_network(data, target_index, num_nodes=20, save_path=None):
    """
    Plot the network graph of the top num_nodes similar nodes for the target.
    
    Parameters:
        data: the dataframe containing the similarity data
        target_index: the index of the target node
        num_nodes: the number of nodes to plot
    """
    # Compute the similarity
    similarity = cosine_similarity(data)
    similarity_df = pd.DataFrame(similarity, index=data.index, columns=data.index)
    
    # Get the top num_nodes similar nodes for the target
    target_similarities = similarity_df.loc[target_index].sort_values(ascending=False)[1:num_nodes+1].index

    # Create a network graph
    G = nx.Graph()

    # Add nodes and edges
    for node1 in target_similarities:
        for node2 in target_similarities:
            if node1 != node2 and similarity_df.loc[node1, node2] > 0.7:
                G.add_edge(node1, node2, weight=similarity_df.loc[node1, node2])
                
        # Add edge from the target node to all similar nodes
        G.add_edge(target_index, node1, weight=similarity_df.loc[target_index, node1])

    # Add the target node
    G.add_node(target_index)
    
    # Define color maps for nodes and edges
    node_color = [i for i in range(G.number_of_nodes())] 
    edge_color = [G[u][v]['weight'] for u,v in G.edges()]

    # Draw the network
    plt.figure(figsize=(10,8))
    pos = nx.spring_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_color, cmap=plt.cm.Blues)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='darkorange', font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, edge_cmap=plt.cm.viridis, width=2, alpha=0.2)  # 20% transparency

    plt.axis('off')  # to turn off the frame
    
    if save_path:
        plt.savefig(save_path)
        
    else:
        plt.show()






# Hexbin
def plot_hexbin(data, x_axis, y_axis, gridsize=20, save_path=None):
    """
    Plot a hexbin plot of the specified columns in the input dataframe.
    
    Parameters:
        data (pandas.DataFrame): The input dataframe.
        x_axis (str): The name of the column to plot on the x-axis.
        y_axis (str): The name of the column to plot on the y-axis.
    """
    sns.jointplot(x=x_axis, y=y_axis, data=data, kind='hex', gridsize=20)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return plt



# Line Heatmap
def plot_line_heatmap(df, name_value, cmap='vlag'):  
    """
    Plot a line plot and a heatmap of the specified row in the input dataframe.
    Heatmap includes all columns except 'Name'.
    
    Parameters:
        df (pandas.DataFrame): The input dataframe.
        name_value (str): The value in the 'Name' column of the row to plot.
        cmap (str): The name of the colormap to use for the heatmap. Defaults to 'vlag'.
    """
    # Find the row where 'Name' is equal to name_value
    row = df.loc[df['Name'] == name_value]

    # Extract the desired columns and convert them to a list
    desired_columns = ['Granger', 'log2FoldChange', 'Pearson', 'Spearman']
    list1 = row[desired_columns].values.tolist()[0]

    # Get the other columns by dropping the ones already in list1
    remaining_columns = df.drop(columns=['Name'] + desired_columns)
    list2 = remaining_columns.loc[remaining_columns.index == row.index[0]].values.tolist()[0]

    # Reshape list1 into a 2D array for the heatmap
    list1_2d = np.array(list1).reshape(-1, 1)

    # Create subplots with adjusted sizes
    fig = plt.figure(figsize=(10, 2))
    grid = plt.GridSpec(1, 10, hspace=0.2, wspace=0.2)  # We'll use 8 columns in total

    # Plot the line graph on the left (using 7 out of 8 columns)
    ax1 = plt.subplot(grid[:9])  # equivalent to grid[0, :7]
    ax1.plot(list2, color='navy')
    ax1.set_title(name_value)
    ax1.set_xticks(range(len(remaining_columns.columns)))
    ax1.set_xticklabels(remaining_columns.columns, rotation=90)

    # Plot the heatmap on the right (using 1 out of 8 columns) without color bar
    ax2 = plt.subplot(grid[9:])  # equivalent to grid[0, 7:]
    sns.heatmap(list1_2d, ax=ax2, cmap=cmap, cbar=False, yticklabels=desired_columns)
    ax2.yaxis.tick_right()
    ax2.xaxis.set_visible(False)  # hide the x-axis
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)  # set rotation to 0

    # Show the plotc
    plt.tight_layout()
    plt.show()



def save_table_as_svg(df, save_path=None):
    """
    Save a pandas dataframe as an svg file.
    """ 
    # Set global font properties
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    fig, ax = plt.subplots(1,1)
    #set figure size
    fig.set_size_inches(12, 5.5)

    cell_text = []
    for row in range(len(df)):
        cell_text.append([f"{val:.5}" if isinstance(val, float) else (str(val)[:66] + '....' if len(str(val)) > 70 else str(val)) for val in df.iloc[row]])

    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=cell_text, colLabels=df.columns, cellLoc='center', loc='center')

    # Autoscale
    table.auto_set_column_width(list(range(len(df.columns))))
    table.scale(1, 1.5)  # may need to adjust based on specific table size

    # Change cell color for header cells and make them bold
    for (row, col), cell in table.get_celld().items():
        if (row == 0):
            cell.set_facecolor("steelblue")  # change color as desired
            cell.get_text().set_color('white')  # change font color as desired
            cell.get_text().set_fontsize(12)  # make headers slightly larger
            cell.get_text().set_weight('bold')  # make headers bold
        elif row % 2 == 0:  # alternate color for non-header cells
            cell.set_facecolor("seashell")
        else:
            cell.set_facecolor("aliceblue")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

   


def plot_result(result, score_column, log2FoldChange_column, Granger_column, save_path=None):
    """
    Plot the result of the result.
    
    Parameters:
    result (pandas dataframe): result of the pipeline
    score_column (str): name of the column for x values
    log2FoldChange_column (str): name of the column for size
    Granger_column (str): name of the column for color
    """
    # Convert categorical column 'Granger' into numerical values for color
    result['color'] = result[Granger_column].astype('category').cat.codes

    # Create a new column for size based on 'log2FoldChange', 
    # you might need to adjust this calculation based on your data
    model = MinMaxScaler()

    #min-max scaling
    result[log2FoldChange_column] = model.fit_transform(result[log2FoldChange_column].values.reshape(-1,1))
    result['size'] = np.abs(result[log2FoldChange_column]) * 150 

    # Get color palette
    palette = sns.color_palette("crest", as_cmap=True)

    fig, ax = plt.subplots()
    
    # Set the figure size
    fig.set_size_inches(15, 7)

    # Adjust subplot parameters to give more space to y-labels
    plt.subplots_adjust(left=0.2)

    # Using 'Score' for x values, 'Name' for y labels, color by 'color', and size by 'size'
    scatter = ax.scatter(x=result[score_column], y=result['Name'], c=result['color'], s=result['size'], cmap=palette, edgecolor='black')

    # Invert y axis to have index 1 at the top and the highest index at the bottom
    ax.invert_yaxis()

    # Create a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(Granger_column)

    plt.xlabel(score_column)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()







# ***************************  Temporarily Unavailable  *******************************
# # Line Heatmap for all data
# def plot_data(df, name_value, ax1, ax2, cmap='vlag'):
#     """
#     plot the line graph and heatmap for the given gene name
    
#     df: dataframe
#     name_value: the name of the gene
#     ax1: the axis for the line graph, ax1 = plt.subplot(grid[:9])
#     ax2: the axis for the heatmap, ax2 = plt.subplot(grid[9:])
#     cmap: the color map for the heatmap
#     """
#     # Find the row where 'Name' is equal to name_value
#     row = df.loc[df['Name'] == name_value]

#     # Extract the desired columns and convert them to a list
#     desired_columns = ['Granger', 'log2FoldChange', 'Pearson', 'Spearman']
#     list1 = row[desired_columns].values.tolist()[0]

#     # Get the other columns by dropping the ones already in list1
#     remaining_columns = df.drop(columns=['Name'] + desired_columns)
#     list2 = remaining_columns.loc[remaining_columns.index == row.index[0]].values.tolist()[0]

#     # Reshape list1 into a 2D array for the heatmap
#     list1_2d = np.array(list1).reshape(-1, 1)

#     # Plot the line graph on the left 
#     ax1.plot(list2, color='navy')
#     ax1.set_title(name_value)
#     ax1.set_xticks(range(len(remaining_columns.columns)))
#     ax1.set_xticklabels(remaining_columns.columns, rotation=90)

#     # Plot the heatmap on the right without color bar
#     sns.heatmap(list1_2d, ax=ax2, cmap=cmap, cbar=False, yticklabels=desired_columns)
#     ax2.yaxis.tick_right()
#     ax2.xaxis.set_visible(False)  # hide the x-axis
#     ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)  # set rotation to 0


# def plot_all_data(df):
#     """
#     plot the line graph and heatmap for the top 10 genes in the dataframe
#     """
#     df = df.head(10)
#     num_rows = df.shape[0]
#     fig = plt.figure(figsize=(10, num_rows * 2))  # adjust the figure height based on the number of rows
#     grid = plt.GridSpec(num_rows, 10, hspace=0.5, wspace=0.2)

#     for i, row in df.iterrows():
#         ax1 = plt.subplot(grid[i, :9])  # line graph
#         ax2 = plt.subplot(grid[i, 9:])  # heatmap
#         plot_data(df, row['Name'], ax1, ax2)
        
#         # Hide the x labels except for the last line plot
#         if i < num_rows - 1:
#             ax1.set_xticklabels([]) 
#         else:
#             ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

#     plt.tight_layout()
#     plt.show()
# ***************************  Temporarily Unavailable  *******************************














# AI
# def Yuanfang(df, target):
#     """
#     Use OpenAI's API to generate a question for the user to answer.
#     Question: Which one may be involved in the synthesis of target?
    
#     Parameters:
#         df: the dataframe containing the similarity data
#         target: the target node
#     """
#     # annotation = read_upload(annotation_file)
#     # df = merge_dataframes([df, annotation])
#     df = df.head(20)
    
#     #if 'Description' in df.columns, is 'Descripton is not in df.columns, then use 'Name'
#     if 'Description' in df.columns:
#         hits = df['Description'].to_list()
#     else:
#         hits = df['Name'].to_list()
#     # hits = df['Description'].to_list()
    
#     hits = [str(item) for item in hits]
#     hits = ', '.join(hits)
    
#     q = hits + '\n\n\nWhich one may be involved in the synthesis of ' + target + '?'
    
#     openai_api_key = getpass.getpass("Please enter your OpenAI API Key: ")
#     openai.api_key = openai_api_key

#     messages = [
#         {"role": "system", "content": "You are a biological chemist and can explain biological mechanisms"},
#         {"role": "user", "content": q}
#     ]

#     completion = openai.ChatCompletion.create(
#         model = "gpt-3.5-turbo",
#         temperature = 0.8,
#         max_tokens = 2000,
#         messages = messages
#     )
    
#     print(' ')
#     print(completion.choices[0].message.content)
#     print(' ')
#     print(' ')
#     print('NOTICE: The output was produced by the large language model GPT 3.5 turbo, so it should only be regarded as a source of inspiration.')

def Yuanfang(df, target, output_path=None):
    """
    Use OpenAI's API to generate a question for the user to answer.
    Question: Which one may be involved in the synthesis of target?
    
    Parameters:
        df: the dataframe containing the similarity data
        target: the target node
        output_path: (optional) path to save the output as a .txt file
    """
    df = df.head(50)
    
    if 'Description' not in df.columns:
        error_message = "Please provide a gene annotation file to use this feature. For how to obtain it, please refer to: http://www.catbridge.work/myapp/tutorial/"
        if output_path:
            with open(output_path, 'w') as file:
                file.write(error_message)
        else:
            print(error_message)
        return

    hits = df['Description'].to_list()
    hits = [str(item) for item in hits]
    hits = ', '.join(hits)
    
    q = hits + '\n\n\nWhich one may be involved in the synthesis of ' + target + '?'
    
    openai_api_key = getpass.getpass("Please enter your OpenAI API Key: ")
    openai.api_key = openai_api_key

    messages = [
        {"role": "system", "content": "You are a biological chemist and can explain biological mechanisms"},
        {"role": "user", "content": q}
    ]

    completion = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        temperature = 0.8,
        max_tokens = 2000,
        messages = messages
    )
    
    output_content = '\n' + completion.choices[0].message.content + '\n\n\nNOTICE: The output was produced by the large language model GPT 3.5 turbo, so it should only be regarded as a source of inspiration.'
    
    if output_path:
        # Save the output to the specified path
        with open(output_path, 'w') as file:
            file.write(output_content)
    else:
        # Print the output
        print(output_content)













"""
#  ██████╗ █████╗ ████████╗    ██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗
# ██╔════╝██╔══██╗╚══██╔══╝    ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝
# ██║     ███████║   ██║       ██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗  
# ██║     ██╔══██║   ██║       ██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝  
# ╚██████╗██║  ██║   ██║       ██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗
#  ╚═════╝╚═╝  ╚═╝   ╚═╝       ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝
"""


def pipeline(gene_file, metabo_file, design_file, annotation_file, target, cluster_count, max_lag=1, aggregation_func=None):
    """
    This function processes gene expression data, performs computations, and returns results.

    Parameters:
    - gene_file (str): The filename of the gene count data.
    - metabo_file (str): The filename of the metabolome data.
    - design_file (str): The filename of the experimental design data (can be None).
    - annotation_file (str): The filename of the gene annotation data (can be None).
    - target (str): The target metabolite for the analysis.
    - cluster_count (int): The number of clusters for time series clustering.
    - max_lag (int): The maximum number of lags for Granger causality test (default is 1).
    - aggregation_func (function): The function to be used for data aggregation.

    Returns:
    - result (pd.DataFrame): A pandas DataFrame containing the processed results, with annotations and clustering information if provided.
    """
    # Read data
    gene = read_upload(gene_file)
    metabo = read_upload(metabo_file)

    if design_file is not None:
        # If there is a design file
        design = read_upload(design_file)
        # Process data
        processed_gene = aggregation_func(gene, design)
        processed_metabo = aggregation_func(metabo, design)
        # Get target data
        t = get_target(target, processed_metabo)
        # Compute Granger causality
        granger = compute_granger(processed_gene, t, max_lag)
        pearson = compute_pearson(processed_gene, t)
        # Prepare dataframe for fold change calculation
        df_for_fc(processed_metabo, target, gene, design)
        # Compute fold change
        fc = fc_comp()
        # Merge data
        data = merge_dataframes([granger, fc, pearson])

    else:
        # If there is no design file
        t = get_target(target, metabo)
        noontide = find_noontide(metabo, target)
        granger = compute_granger(gene, t, max_lag)
        # Compute fold change
        fc = no_repeat_fc(gene, noontide)
        data = merge_dataframes([granger, fc])
    
    # # If there is an annotation file
    # if annotation_file is not None:
    #     annotation = read_upload(annotation_file)
    #     data = merge_dataframes([data, annotation])
    #     data = annotation_score(data)  # Assuming annotation_score modifies the data based on annotation to be involved in the final score
    
    # result = score(data)
    # # Perform clustering
    # cluster = ts_clustering(gene, cluster_count)
    # result = merge_dataframes([result, cluster])
    # result.set_index('Rank', inplace=True)

    # return result
    if annotation_file is not None:
        annotation = read_upload(annotation_file)
        data = merge_dataframes([data, annotation])
        data = add_annotation_score(data)  # Add a new column for annotation score

    # result = compute_score(data)
    # # Perform clustering
    # cluster = ts_clustering(gene, cluster_count)
    # result = merge_dataframes([result, cluster])
    # result.set_index('Rank', inplace=True)
    data['log2FoldChange'].fillna(0, inplace=True)
    data['Granger'].fillna(0, inplace=True)
    # add a new column ['Score'] to the dataframe
    # use minmax scaling Granger and log2FoldChange before Euclidean distance
    model = MinMaxScaler()
    data['Score'] = model.fit_transform(data['Granger'].values.reshape(-1,1)) + model.fit_transform(data['log2FoldChange'].values.reshape(-1,1))

    
    
    #min-max scaling
    data['Score'] = model.fit_transform(data['Score'].values.reshape(-1,1))
    result = data.sort_values(by='Score', ascending=False)
    # add a new column ['Rank'] to the dataframe
    result['Rank'] = np.arange(1, len(result)+1)
    result.set_index('Rank', inplace=True)

    return result





def compute_corr(
    gene_file, metabo_file, design_file, annotation_file,
    target, cluster_count, aggregation_func=None, 
    lag=1, E=3, tau=1, n_components=1
):
    """
    Computes correlation metrics between gene expression and metabolite data.
    
    This function reads in gene expression data, metabolite data, optional experimental design, 
    and annotation data from specified files. It then processes the data, computes various correlation 
    and similarity metrics, performs clustering, and optionally annotates the results, 
    returning a DataFrame containing the aggregated results.
    
    Parameters:
    - gene_file (str): Path to the file containing gene count data.
    - metabo_file (str): Path to the file containing metabolome data.
    - design_file (str): Optional; Path to the file containing experimental design data. Default is None.
    - annotation_file (str): Optional; Path to the file containing gene annotation data. Default is None.
    - target (str): The target metabolite for analysis.
    - cluster_count (int): Number of clusters for time series clustering.
    - aggregation_func (callable, optional): Function for data aggregation. Default is None.
    - lag (int, optional): Maximum number of lags for Granger causality test. Default is 1.
    - E (int, optional): The embedding dimension for Convergent Cross Mapping (CCM). Default is 3.
    - tau (int, optional): The delay for Convergent Cross Mapping (CCM). Default is 1.
    - n_components (int, optional): Number of components for Canonical Correlation Analysis (CCA). Default is 1.
    
    Returns:
    - result (pd.DataFrame): A DataFrame containing the computed metrics, clustering results, 
                             and optionally annotations.
    
    Raises:
    - Various exceptions may be raised due to file reading, data processing, or computation errors.
    
    Notes:
    - The aggregation_func parameter should be a function that takes two arguments: a DataFrame 
      containing data and a DataFrame containing design information, and returns a processed DataFrame.
    - If the design_file is provided, the data will be aggregated based on the provided aggregation_func.
    - The function employs a variety of metrics such as Granger causality, Convergent Cross Mapping (CCM),
      Canonical Correlation Analysis (CCA), Dynamic Time Warping (DTW), Cross-Correlation Function (CCF),
      Spearman correlation, and Pearson correlation to analyze the relationships between gene expression 
      and metabolite data.
    - It performs time series clustering on the gene expression data.
    - If the annotation_file is provided, the function will merge the annotation data with the results, 
      and compute an annotation score.
    """
    # Read data
    gene = read_upload(gene_file)
    metabo = read_upload(metabo_file)
    
    if design_file is not None:
        design = read_upload(design_file)
        # Process data
        processed_gene = aggregation_func(gene, design)
        processed_metabo = aggregation_func(metabo, design)
        t = get_target(target, processed_metabo)
        df_for_fc(processed_metabo, target, gene, design)
        fc = fc_comp()
        
    else:
        t = get_target(target, metabo)
        noontide = find_noontide(metabo, target)
        processed_gene = gene
        processed_metabo = metabo
        fc = no_repeat_fc(gene, noontide)
    
    # Compute corr
    granger_score = compute_granger(processed_gene, t, maxlag=lag)
    ccm_score = compute_ccm(processed_gene, t, E=3, tau=1)
    cca_score = compute_cca(processed_gene, t, n_components=1)
    dtw_score = compute_dtw(processed_gene, t)
    ccf_score = compute_ccf(processed_gene, t, lag=1)
    spearman_score = compute_spearman(processed_gene, t)
    pearson_score = compute_pearson(processed_gene, t)
    clustering_result = ts_clustering(gene, cluster_count)


    data = merge_dataframes([granger_score, ccm_score, cca_score, dtw_score, ccf_score, spearman_score, pearson_score, fc, clustering_result])


    if annotation_file is not None:
        annotation = read_upload(annotation_file)
        data = merge_dataframes([data, annotation])
        data = add_annotation_score(data)  # Add a new column for annotation score

    return data





"""
#                                    ___
#                       |\---/|  / )|Guo|
#           ------------;     |-/ / |Lab|
#                       )     (' /  `---'
#           ===========(       ,'==========
#           ||  _      |      |      
#           || ( (    /       ;
#           ||  \ `._/       /
#           ||   `._        /|
#           ||      |\    _/||
#         __||_____.' )  |__||____________
#          ________\  |  |_________________
#                   \ \  `-.
#                    `-`---'  
"""

"""                
#  ██████╗ █████╗ ████████╗    ██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗
# ██╔════╝██╔══██╗╚══██╔══╝    ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝
# ██║     ███████║   ██║       ██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗  
# ██║     ██╔══██║   ██║       ██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝  
# ╚██████╗██║  ██║   ██║       ██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗
#  ╚═════╝╚═╝  ╚═╝   ╚═╝       ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝
"""
