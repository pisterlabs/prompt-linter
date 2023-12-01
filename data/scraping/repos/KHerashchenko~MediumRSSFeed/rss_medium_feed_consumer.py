from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import from_json, udf
from pyspark.sql.types import StructType, StructField, ArrayType, StringType
import argparse
import requests
from bs4 import BeautifulSoup
import openai
import os


def parse_medium_story_text(link):

    response = requests.get(link)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all('p', {'class': 'pw-post-body-paragraph'})
    top_two_paragraphs = [p.get_text().strip() for p in paragraphs[:2]]

    story_text = '\n\n'.join(top_two_paragraphs)

    return story_text


def fetch_tags(link):
    # Send an HTTP GET request to the URL
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    div_tags = soup.find_all('div', class_="pi go cw pj ed pk pl be b bf z bj pm")
    list_of_tags = [tag.get_text() for tag in div_tags]

    return list_of_tags


def summarise_desc(description_text):

    # OpenAI secret key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if description_text == 'Not provided':
        return description_text

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{'role': 'system', 'content': 'Summarize content you are provided in 1 sentence.'},
                  {"role": "user", "content": description_text}],
        temperature=0,
        max_tokens=64
    )

    return response['choices'][0]['message']['content']


# Define and parse the command-line arguments
parser = argparse.ArgumentParser()

# Kafka configuration
KAFKA_BROKER = 'localhost:9092'

# MongoDB configuration
MONGODB_CONNECTION_STRING = "mongodb://node1:27017/"  # Your MongoDB connection string
MONGODB_DB_NAME = "rss_db"
MONGODB_COLLECTION_NAME = 'general'


def run_spark_consumer_app(topic):

    # Create a SparkSession
    spark = SparkSession.builder.appName("KafkaMediumRSSConsumer")\
        .config("spark.mongodb.input.uri", "mongodb://node1/rss_db.general")\
        .config("spark.mongodb.output.uri", "mongodb://node1/rss_db.general")\
        .getOrCreate()

    schema = StructType([
        StructField("title", StringType(), True),
        StructField("link", StringType(), True),
        StructField("description", StringType(), True),
        StructField("published", StringType(), True),
        StructField("tag", StringType(), True),
    ])

    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", topic) \
        .load()

    json_df = kafka_df.selectExpr("CAST(value AS STRING) as json") \
        .select(from_json("json", schema).alias("data")) \
        .select("data.*")

    parse_medium_story_text_udf = udf(parse_medium_story_text, StringType())
    link_column = json_df['link']
    json_df_with_desc = json_df.withColumn("description", parse_medium_story_text_udf(link_column))

    summarise_description_udf = udf(summarise_desc, StringType())
    desc_column = json_df_with_desc['description']
    json_df_with_desc_sum = json_df_with_desc.withColumn("summary", summarise_description_udf(desc_column))

    # Print the received events to the console
    query_console = json_df_with_desc_sum.selectExpr("CAST(tag AS STRING)", "CAST(title AS STRING)", "CAST(published AS STRING)") \
        .writeStream \
        .outputMode("append") \
        .format("console") \
        .start()

    dataStreamWriter = json_df_with_desc_sum.writeStream \
        .format("mongodb") \
        .option("checkpointLocation", "checkpoint")\
        .option("spark.mongodb.connection.uri", MONGODB_CONNECTION_STRING) \
        .option("spark.mongodb.database", MONGODB_DB_NAME) \
        .option("spark.mongodb.collection", MONGODB_COLLECTION_NAME) \
        .outputMode("append")

    # Start the MongoDB query
    query_mongodb = dataStreamWriter.start()

    # Await termination for both queries
    query_console.awaitTermination()
    query_mongodb.awaitTermination()


if __name__ == '__main__':
    parser.add_argument('--topic', type=str, help='Kafka topic', default='general')
    args = parser.parse_args()

    run_spark_consumer_app(args.topic)
