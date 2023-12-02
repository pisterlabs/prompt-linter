# External packages
import json
from datetime import datetime
import os
import openai
import traceback
import time

#Internal packages
from config import OBJECT_STORAGE_REGION, OBJECT_STORAGE_BUCKET
import spaces_upload
import database
import formata
import web_scrape 
import mongo_db_connector as mongo
import key_word_matcher as keyword
import doc_summarization

keywords_master = database.getkeywords()
#orgs_master = database.getorgs()


def error_logger(e, url):
    with open("error_log_summarization.txt", "a") as file:
        file.write(f"Invalid request error retrieving: {url}")
        file.write("Here is the error:\n")
        file.write(traceback.format_exc())
        file.write("\n")

def write_json_to_file(json_data, filename, directory):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Construct the full file path
    filepath = os.path.join(directory, filename)

    # Write the JSON data to a .json file
    with open(filepath, 'w') as file:
        json.dump(json_data, file, indent=4)

def refine(orgid,url,page_text):
    #step 1: create doc summary
    try:
        summary = doc_summarization.summarize_doc(document=page_text)        
    except openai.error.InvalidRequestError as e:
        summary = "Failed to create the summary file for this website."
        error_logger(e, url=url)
    except Exception as e:
        summary = "Failed to create the summary file for this website."
        error_logger(e, url=url)
    #step 2: get keywords
    try:
        keywords = keyword.count_keywords(text=page_text,keywords=keywords_master)
    except:
        keywords = 'Failed to detect keywords'
    #step 3: get org references
    #try:
    #    reference_orgs = keyword.count_keywords(text=summary,keywords=orgs_master)
    #except:
    #    reference_orgs = 'Failed to detect keywords'
    
    finally:
        #step 4: format updates to push
        updates = {
            'summary':summary,
            'keywords':keywords,
            'refined':True,
            'lastupdate':datetime.utcnow()
        }
        #step 3: push updates to mongodb source
        try:
            mongo.updateitem(database='scraped',collection=orgid,item_field='page_url',item_value=url,update_file=updates)
            return(True)
        except:
            return(False)
        
def run(domains: 'list' = None):
    #step 1: get orgid from input domain
    orgids = [{'domain':item,'orgid':database.getorgid(item)} for item in domains]

    #step 2: get pages from domain that need processing
    need_refining = [mongo.get_pages_to_refine(database='scraped',collection=item['orgid']) for item in orgids]

    #step 3: summarize pages that have page data but processed set to false
    index = 0
    for item in need_refining:
        for record in item:
            print(record['page_url'])
            refine(orgid=orgids[index]['orgid'],url=record['page_url'],page_text=record['page_text'])            
        index = index + 1
    
    #step 4: Add pageid, detect keywords

#example
#print(run(domains=['aws.amazon.com']))

"""
with open ('websites.txt', 'rt') as myfile:  # Open websites.txt for reading
    for myline in myfile:              # For each line, read to a string,
        url = myline.strip()        # Each line is a new URL to open and scrape
        start_time = time.time() # Start the timer to capture how long entire process takes. 
        
        # Step 1: Webpage Scraping
        page_text = web_scrape.extract_text_from_url(url)      
        url_cleaned = formata.clean_page(url)
        page_data = database.getpagedetails(url_cleaned)
        keywords = database.getkeywords()

        if page_data is None:
            page_data = {'pid':'NOT_FOUND/'+url_cleaned.replace("/","_"),'did':'domainid not found','orgid':'orgid not found'}
        if page_text:
            
            # Step 2: Data Transformation
            transformed_data = spaces_upload.transform_data(page_text)

            # Step 3: Storing Data in DigitalOcean Spaces
            object_name = str(page_data['pid'])
            orgid = str(page_data['orgid'])
            domainid = str(page_data['did'])
            today = str(date.today())
            # Set up the metadata to attach to the spaces storage
            metadata = {'URL': url, 'Ingestion_Date': today,'URL_cleaned':url_cleaned,'orgid':orgid,'domainid':domainid}
            spaces_upload.upload_to_spaces(bucket_name, object_name, transformed_data, 
                s3config["endpoint_url"], s3config["aws_access_key_id"], 
                s3config["aws_secret_access_key"], metadata = metadata)

            # Step 4: Get the keywords in the Website. Add in pid, orgid, and did. 
            keyword_JSON = key_word_matcher.count_keywords(page_text, keywords)
            keyword_JSON['URL']=url #Adding the URL to the JSON output
            keyword_JSON['PID']=object_name #Adding the PID to the output
            keyword_JSON['orgid']=orgid #Adding the Organization ID to the output
            keyword_JSON['did']=domainid #Adding the Domain ID to the output
            
            # Try to create the summary for a given document. If this fails, document it
            try:
                keyword_JSON["Summary"] = doc_summarization.summarize_doc(document=page_text)
            except openai.error.InvalidRequestError as e:
                keyword_JSON["Summary"] = "Failed to create the summary file for this website."
                error_logger(e, url=url)
            except Exception as e:
                keyword_JSON["Summary"] = "Failed to create the summary file for this website."
                error_logger(e, url=url)

            #print(today)
            json_output = json.dumps(keyword_JSON, indent=4)
            mongo.send_json_to_mongodb(json_data=json_output,orgid=orgid)

            # Create a summary for each webpage and write to JSON

            
            end_time = time.time()

            execution_time = end_time - start_time
            print(f"The website {url} took {execution_time} seconds to create.")



ARCHIVED
s3config = {
    "region_name": OBJECT_STORAGE_REGION,
    "endpoint_url": "https://{}.digitaloceanspaces.com".format(OBJECT_STORAGE_REGION),
    "aws_access_key_id": OBJECT_STORAGE_KEY,
    "aws_secret_access_key": OBJECT_STORAGE_SECRET }

bucket_name = OBJECT_STORAGE_BUCKET
"""