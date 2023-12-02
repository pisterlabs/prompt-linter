import os
import json
import time
import openai
import googlemaps
import pandas as pd
from sklearn.cluster import KMeans
from access_credentials.gcp_credentials import GOOGLE_MAPS_KEY

openai.api_key = os.environ.get('OPENAI_API_KEY')

dir_path = os.path.dirname(os.path.realpath(__file__))

SETTINGS_PROMPT = """
You are a helpful assistant. Your task is to find the names of the cities or towns affected by a natural disaster.

a. Input Description: 

The input will contain 1. The type of natural disaster that occured 2. Which year it occured 3. Which country and continent it occured in.

b. Task Description:

1. Find the names of City/Cities/Town/Towns where the natural disaster affected.
3. And strictly return only the names of cities or towns seperated by comman.

c. Response Example: return City_1, City_2, ..., City_n

d. In case no information is found just return "Not Found". No need to give explanations on why information was not found.
"""


def extract_lat_lon(df: pd.DataFrame) -> None:
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_KEY)

    for row in df.iterrows():
        dis_no = row[1]['DisNo.']
        file_path = os.path.join(dir_path, f'../Datasets/IntermediateDatasets/GeocodedJsonFiles/{dis_no}.json')

        if not os.path.exists(file_path):
            try:
                geocode_result = gmaps.geocode(f"({row[1]['Location']}) in {row[1]['Country']}")
                with open(file_path, 'w') as file:
                    file.write(json.dumps(geocode_result))
            except Exception as e:
                print(e)


def extract_missing_locations(df: pd.DataFrame) -> None:
    try:
        try:
            file_path = os.path.join(dir_path, f'../Datasets/DB_CSVs/location_filled.csv')
            location_dataframe = pd.read_csv(file_path, sep="|")
        except Exception as e:
            print(e)
            location_dataframe = df[df.Location.isna()][['DisNo.', 'Start Year', 'Disaster Type',
                                                         'Country', 'Location']]
            file_path = os.path.join(dir_path, f'../Datasets/DB_CSVs/location_filled.csv')
            location_dataframe.to_csv(file_path, sep="|", index=False)

        if len(location_dataframe[location_dataframe.Location.isna()]) == 0:
            return

        file_path = os.path.join(dir_path, f'../Datasets/DB_CSVs/location_filled.csv')
        replacement_df = pd.read_csv(file_path, sep="|")
        for i in location_dataframe[location_dataframe.Location.isna()].iterrows():
            prompt = f"1. Type: {i[1]['Disaster Type']}.\n2. Year {i[1]['Start Year']}.\n3. {i[1]['Country']}, Europe."
            replacement_df.loc[replacement_df['DisNo.'] == i[1]['DisNo.'], 'Location'] = get_cities_and_towns(prompt)
            replacement_df.to_csv(file_path, sep="|", index=False)
            time.sleep(3)
    except Exception as e:
        print(e)
        extract_missing_locations(df)


def get_cities_and_towns(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                'role': 'system',
                'content': SETTINGS_PROMPT
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        temperature=0.0
    )

    return response.choices[0]['message']['content']


def cluster_similar_locations():
    disaster = pd.read_csv('../Datasets/CleanedDatasets/Disaster.csv', sep='|')
    disaster_clf = pd.read_csv('../Datasets/CleanedDatasets/DisasterClassification.csv', sep='|')
    location_df = pd.read_csv('../Datasets/CleanedDatasets/Location.csv', sep='|')

    merged_df = pd.merge(disaster, disaster_clf, on='ClassificationKey', how='inner')
    merged_df = merged_df[merged_df['Type'] == 'Extreme temperature']
    merged_df = pd.merge(merged_df, location_df, left_on='DisasterNum', right_on='DisasterNo', how='inner')

    locations = merged_df[['Latitude', 'Longitude']].values
    num_clusters = 100
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(locations)

    merged_df['Cluster'] = kmeans.labels_

    merged_df = merged_df[['DisasterNum', 'ClassificationKey', 'ISOCode',
                           'Location', 'Latitude', 'Longitude', 'Cluster']]

    cluster_centers = kmeans.cluster_centers_

    centroids_df = pd.DataFrame(cluster_centers, columns=['Latitude', 'Longitude'])
    centroids_df['Cluster'] = range(num_clusters)

    merged_df.to_csv('../Datasets/CleanedDatasets/LocationClustersConnection.csv', sep='|', index=False)
    centroids_df.to_csv('../Datasets/CleanedDatasets/LocationClusters.csv', sep='|', index=False)


if __name__ == '__main__':
    cluster_similar_locations()
