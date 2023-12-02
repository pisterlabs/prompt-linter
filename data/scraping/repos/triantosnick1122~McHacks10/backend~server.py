import cohere
from cohere.classify import Example
import datetime
import random
import generalUtils
import dbUtils
import prawUtils
import csv
from typing import List


cohereClient = cohere.Client(generalUtils.getCohereApiKey())


def get_examples(dataset_percentage: float = 1, random_state: int = 13) -> cohere.classify.Example:
    """Read toxicity dataset entries and loads it up as a list of examples."""
    assert dataset_percentage >= 0 and dataset_percentage <= 1, "Percentage must be between 0 and 1."
    random.seed(random_state)
    toxic_examples = []
    non_toxic_examples = []
    
    with open("../data/Social Media Toxicity Dataset.csv", "r", encoding="utf-8") as toxicity_dataset:
        reader = csv.DictReader(toxicity_dataset)

        for row in reader:
            if row["Is this text toxic?"] == "Toxic":
                toxic_examples.append(Example(row["text"], "Toxic"))
            else:
                non_toxic_examples.append(Example(row["text"], "Not Toxic"))

    num_entries = int(len(toxic_examples) * dataset_percentage)
    
    examples = random.sample(toxic_examples, num_entries) + random.sample(non_toxic_examples, num_entries)

    return examples


def get_classifications(inputs: List[str]):
    """Output the model's predictions on whether inputs are toxic or not."""
    assert len(inputs) > 0, "Must have at least one input."
    response = cohereClient.classify(  
        model="large",  
        inputs=inputs,  
        examples=get_examples(0.1)
    )
    return response.classifications


def get_subreddit_toxicity(subreddit_name: str, sort_type: str, number_requests : int): # Ideally would be a subreddit name here but for now it will be inputs
    """
    Find the toxicity value of the specified subreddit.
    Adds 96 since that is the maximum number of inputs accepted in one request to cohere API.
    """
    # Launch reddit API and retrieve list of text inputs
    print("Collecting reddit posts...")
    inputs = prawUtils.get_posts(subreddit_name, sort_type , number_requests)
    print("Done collecting reddit posts...")
    start_index = 0
    total_toxicity_val = 0
    while start_index < len(inputs):
        print("Making a request to Cohere API...")
        classifications = get_classifications(inputs[start_index:start_index + 96])
        for classification in classifications:
            total_toxicity_val += classification.labels["Toxic"].confidence
        start_index += 96
    
    return total_toxicity_val / len(inputs)



"""Saves a generated report to the db"""
# def saveGeneratedReport(subreddit, timestamp, score, records_analyzed, is_current, is_post):
def saveGeneratedReport(subreddit, score, num_requests, sort_type):
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    dbUtils.executeInsertOrUpdate(dbUtils.generateFullInsertStmt(subreddit, timestamp, score, num_requests, sort_type, 1, 1))    
    setAllReportsNotCurrentExceptForNewest(subreddit, 1)

def getAllGeneratedReports():
    return dbUtils.select_query('SELECT * from report;')

def getGeneratedReport(id):
    return dbUtils.select_query('SELECT * from report where id = ' + str(id) + ";")

def getAllCurrentGeneratedReports():
    return dbUtils.select_query("SELECT * from report where is_current = 1;")

def getAllGeneratedReportsForSubreddit(sub_name):
    return dbUtils.select_query("SELECT * from report where subreddit = '" + sub_name + "';")

def getAllCurrentGeneratedReportsForSubreddit(sub_name):
    return dbUtils.select_query("SELECT * from report where subreddit = '" + sub_name + "' AND is_current = 1;")

def setAllReportsNotCurrentExceptForNewest(sub_name, is_post):
    newestId = getNewestReportForSubreddit(sub_name, is_post).id
    setAllReportsNotCurrentExceptForOne(sub_name, is_post, newestId)  

def setAllReportsNotCurrentExceptForOne(sub_name, is_post, idOfOneToKeepCurrent):
    dbUtils.executeInsertOrUpdate(dbUtils.generateUpdateStmtToSetNotCurrent(sub_name, is_post, idOfOneToKeepCurrent))

def getNewestReportForSubreddit(sub_name, is_post):
    query = "SELECT * from report where subreddit = '" + sub_name + "' AND is_post = " + str(is_post) + "\n ORDER BY timestamp DESC;"    
    print(query)
    # print (dbUtils.select_query(query))[0]
    return dbUtils.select_query(query)[0]
    

# TESTING ================================================================

'''
from time import time
start = time()
print(get_subreddit_toxicity("politics"))
end = time()
print(f"Time taken: {end - start}")
'''
