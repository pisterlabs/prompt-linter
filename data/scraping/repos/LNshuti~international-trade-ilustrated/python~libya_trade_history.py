import numpy as np 
import pandas as pd 
import openai 
import pyarrow.parquet as pq
import matplotlib as mpl
import matplotlib.pyplot as plt


def load_data():     

    # Read in the data from parquet 
    labelled_df = pq.ParquetDataset('app/processed_country_partner_df.parquet').read_pandas().to_pandas()

    #labelled_df = pd.read_csv("processed_country_partner_df.csv")
    return labelled_df
    


def main(): 
    data = load_data()

    
    data = data[data['Country Name'] == "Libya"]

    print(data.head())


if __name__ == '__main__':
    main()
