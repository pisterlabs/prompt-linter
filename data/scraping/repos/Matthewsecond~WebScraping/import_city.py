import sqlalchemy
from GeoProcessing import get_kraj, get_latitude, get_longitude
import pandas as pd
import geopy
import unicodedata
from pandasai import SmartDataframe, SmartDatalake
from pandasai.llm import OpenAI
import pandasai

engine = sqlalchemy.create_engine(
    'mysql+pymysql://admin:N6zmVKVW@jobs-intelligence-slovakia.'
    'cluster-c0rbbiliflyo.eu-central-1.rds.amazonaws.com:9906/General_Intelligence')

conn = engine.connect()


def drop_table(table_name):
    try:
        query = sqlalchemy.text(f'DROP TABLE {table_name}')
        conn.execute(query)
    except Exception as e:
        print(e)

def get_zipcode(df, geolocator, lat_field, lon_field):
    #df.apply(get_zipcode, axis=1, geolocator=geolocator, lat_field='Lat', lon_field='Lon')
    location = geolocator.reverse((df[lat_field], df[lon_field]))
    try:
        return location.raw['address']['postcode']
    except Exception as e:
        return None


if __name__ == '__main__':
    llm = OpenAI(api_token='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')

    #read webcrawlresults from database
    webcrawlresults = pd.read_sql_table('SK_WebCrawlResults', conn)

    #read cities from database
    cities = pd.read_sql_table('Cities_Processed', conn)

    #webcrawlresults = webcrawlresults[:60000]


    dl = SmartDatalake([cities, webcrawlresults], config={"llm": llm})

    #dl = SmartDataframe(df=pd.DataFrame(webcrawlresults), config={"llm": llm})
    #response = dl.chat('merge cities in cities dataframe with locality in webcrawlresults. Ignore small differences in text. Return dataframe.')
    response = dl.chat('merge column municipality in cities dataframe with column location in webcrawlresults.')
    dataframe1 = response.dataframe

    dl = SmartDatalake([cities, webcrawlresults], config={"llm": llm})
    response = dl.chat('merge column city in cities dataframe with column location if there in webcrawlresults.')
    dataframe2  = response.dataframe

    # 'concatenate dataframes dataframe and dataframe2. Compare column link. If it already exists in dataframe1, do not add it.'
    dataframe3 = pd.concat([dataframe1, dataframe2]).drop_duplicates(subset=['link'], keep='first')
    dataframe3.to_sql(name='SK_WebCrawlResults_Processed_test', con=engine, if_exists='append', index=False)

    #dl.chat('Find all matches between column location from webcrawlresults and Municipality from cities and merge the dataframes based on those matches. ''Ignore small differences in text.')


    ## drop column City from cities
    #municipality = cities.drop(columns=['City'])

    #convert column location to unicode
    #webcrawlresults['location'] = webcrawlresults['location'].apply(lambda val: unicodedata.normalize('NFKD', str(val)).encode('ascii', 'ignore').decode())



    #merge webcrawlresults and ciities based on column location from webcrawlresults and column City and Municipality from cities
    #merged2 = webcrawlresults.merge(cities, left_on='location', right_on='Municipality', how='left')

















    '''
    geolocator = geopy.Nominatim(user_agent='123')

    # read cities from database
    cities = pd.read_sql_table('Cities_Processed', conn)

    cities['ZIP_CODE'] = cities.apply(get_zipcode, axis=1, geolocator=geolocator, lat_field='latitude', lon_field='longitude')

    #drop table cities
    drop_table('Cities_Processed')

    #import into Cities_Processed
    cities.to_sql(name='Cities_Processed', con=engine, if_exists='append', index=False)
    '''



    #read cities from database
    #cities = pd.read_sql_table('Cities_Processed', conn)
    #cities = cities[:10]
    #cities['zip_code'] = cities.apply(get_zipcode, axis=1, geolocator=geolocator, lat_field='latitude', lon_field='longitude')
    #print(cities)

    #drop table cities

    #read csv into dataframe, ; is separator
    #add columns to cities_add

    #read csv into dataframe, ; is separator

    #columns=['Municipality','City','Region', 'latitude', 'longitude']

    #cities_add = pd.read_csv(r'C:\Users\labus.INTERCONNECTION\Desktop\WebScrapingProject\Geolocation\cities.csv', names=columns, sep=';' , skiprows=2)

    #cities_add.to_sql(name='Cities_Processed', con=engine, if_exists='append', index=False)




    #cities_add['kraj'] = cities_add['City'].apply(get_kraj)
    #cities_add['latitude'] = cities_add['City'].apply(get_latitude)
    #cities_add['longitude'] = cities_add['City'].apply(get_longitude)
