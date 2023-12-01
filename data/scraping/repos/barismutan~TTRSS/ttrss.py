import re
import json
import requests
import argparse
from bs4 import BeautifulSoup
import openai
from selenium import webdriver
from requests_html import HTMLSession
import requests
import schedule
import logging
import traceback
import ast
from datetime import datetime
import time
from ttrss_errors import *

class TTRSS:

    def __init__(self,config):
        self.GPT_API_KEY=config['GPT_API_KEY']
        self.gpt_endpoint=config['gpt_endpoint']
        self.endpoint=config['ttrss_url']
        self.user=config['ttrss_user']
        self.password=config['ttrss_password']
        self.gpt_config=config['gpt_config']
        # self.mrkdwn_template=config['mrkdwn_template']
        self.zapier_webhook=config['zapier_webhook']
        # self.message_time=config['message_time']

        self.UNREAD_BODY={
            "op":"getHeadlines",
            "feed_id":"0",
            "view_mode":"unread",
            "is_cat":"1"
        }



        self.MARK_AS_READ_BODY={
            "op":"updateArticle",
            "article_ids":None,
            "mode":0,
            "field":2,
            "sid":""
        }#NOTE: field 2 is the "unread" field, setting it to 0 <false> ideally marks it as read

        self.MARK_AS_UNREAD_BODY={
            "op":"updateArticle",
            "article_ids":None,
            "mode":1,
            "field":2,
            "sid":""
        }

        self.EXTERNAL_HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Set-Fetch-Site': 'none',
            'Accept-Encoding': 'gzip, deflate',
            'Set-Fetch-Mode': 'navigate',
            'Sec-Fetch-Dest': 'document',

            
        }

        self.scoring_metric=config['scoring_metric']
        self.total_score=sum([vals for vals in self.scoring_metric["metric"].values()])
        
        with open(config['prompt_file'],'r') as f:
            self.prompt=f.read()
        
        # with open(config['question_file'],'r') as f:
        #     self.question=f.read()
        
        with open(config['mrkdwn_template'],'r') as f:
            self.mrkdwn_template=f.read()
        
        with open(config['country_region_mapping'],'r') as f:
            self.country_region_mapping=json.load(f)
        
        with open(config['region_webhook_mapping'],'r') as f:
            self.region_webhook_mapping=json.load(f)
        
        
        
        self.anomalies_file=open(config['anomalies_file'],'a')
        self.anomalies_file.write("----beginning of new run at "+str(datetime.now())+"--------\n")

        self.FULL_CONTENT= 1
        self.SUMMARY_CONTENT=2

        
        self.session_id=self.login()
        openai.api_key=self.GPT_API_KEY

        self.extract_link_callbacks=[
            self.get_read_more_href,
            self.get_last_body_href_generic
            ]
        
        self.preprocess_html_callbacks=[
            self.remove_head,
            self.remove_scripts,
            self.remove_styles,
            self.remove_header,
            self.remove_footer,
            self.remove_navbar,
            self.remove_ads,
            
            self.remove_meta
            ]
        
        ##test specific config
        self.test_mode=config['test_mode']
        if self.test_mode:

            self.test_size=config['test']['test_size']
            self.test_zapier=config['test']['test_zapier']
            self.test_use_cheap_model=config['test']['test_use_cheap_model'] # better -> write this in the config file
            self.test_mark_as_read=config['test']['test_mark_as_read']

        #TODO: i already wrote specific one for datareaches.net, might as well include it later...
        # self.scoring_metric=config['scoring_metric']




##-----------------TTRSS API calls-----------------##

    def login(self):
        body = {
            "op": "login",
            "user": self.user,
            "password": self.password
        }
        response=requests.get(self.endpoint,data=json.dumps(body))
        response_content=response.json()['content']
        

        return response_content['session_id']

    def get_article(self,article_id):
        body = {
            "op": "getArticle",
            "sid": self.session_id,
            "article_id": article_id
        }
        response=requests.get(self.endpoint,data=json.dumps(body))
        return response.json()['content'][0] #they put the thing in an array for some reason

    def get_headlines(self,category):
        body=self.UNREAD_BODY
        body['feed_id']=category
        response=requests.get(self.endpoint,data=json.dumps(
                body
                ))
        
        return response.json()['content']
    
    def mark_as_read(self,article_id): #idea: we can make this a class.
        if self.test_mode and not self.test_mark_as_read:
            return
        mark_as_read_body=self.MARK_AS_READ_BODY
        mark_as_read_body['article_ids']=article_id
        mark_as_read_body['sid']=self.session_id

        response=requests.post(self.endpoint,data=json.dumps(
            mark_as_read_body
                                            ))
        
        return response
    
    def mark_as_unread(self,article_id):
        mark_as_unread_body=self.MARK_AS_UNREAD_BODY
        mark_as_unread_body['article_ids']=article_id
        mark_as_unread_body['sid']=self.session_id

        response=requests.post(self.endpoint,data=json.dumps(
            mark_as_unread_body
                                            ))

        return response

##-----------------TTRSS API calls-----------------##



##-----------------GPT calls-----------------##
    def make_single_gpt_query(self,prompt):
        
        model = "gpt-3.5-turbo" if self.test_mode and self.test_use_cheap_model else self.gpt_config['model'] #using cheaper model for testing purposes... #NOTE: use even a cheaper model later...
        messages=self.gpt_config['messages']
        messages[0]['content']=prompt
        return openai.ChatCompletion.create(
            model=model,
            messages=messages,
            timeout=self.gpt_config['timeout']
    )


    def gpt_query(self,article):
        prompt=self.prompt+article

        logging.debug("Querying GPT-3.5 Turbo.")
        completion = self.make_single_gpt_query(prompt)
        
        try:
            completion_dict=ast.literal_eval(str(completion.choices[0]['message']['content']))
        except SyntaxError as e:
            logging.error("Syntax error (Probably caused by GPT response.):"+str(e))
            logging.error("GPT response:"+str(completion))
            while True:
                try:
                    completion = self.make_single_gpt_query(prompt)
                    completion_dict=ast.literal_eval(str(completion.choices[0]['message']['content']))
                    break
                except SyntaxError as e:
                    logging.error("Syntax error (Probably caused by GPT response.):"+str(e))
                    logging.error("GPT response:"+str(completion))
                    article=self.trim_text(article)
                    prompt=self.prompt+article ##NEW: trim the article and try again in case of invalid json.
                    continue
            return completion_dict
        # completion_dict=json.loads(str(completion.choices[0]['message']['content']))
        return completion_dict

    def score_gpt_response(self,gpt_response):
        score=0
        metric_keys=self.scoring_metric["metric"].keys()
        if gpt_response==None:
            return -1 #indicates error
        for field in gpt_response.keys():
            response_field=gpt_response[field]
            if type(response_field)==list:
                if len(response_field)==0 or (len(response_field)==1 and response_field[0]=="N/A"):
                    continue
            if gpt_response[field]!="N/A"  and field in metric_keys:
                score += self.scoring_metric["metric"][field]

        return score

##-----------------GPT calls-----------------##









##-----------------Link Extractions-----------------## --> #TODO: move these to a different file

    def extract_article_link_from_summary(self,summary_link):

        response=self.make_request_with_session(summary_link)
        if response==None:
            raise NoLinksFoundException(summary_link) #TODO: change this to a different exception
        html=response.content.decode('utf-8') if response.status_code==200 \
                                            else self.invoke_selenium(summary_link)
        # 
        html=self.preprocess_html(html)

        for callback in self.extract_link_callbacks:
            href=callback(html)
            if href is not None:
                
                href=self.remove_excess_whitespace(href)
                return href
        
        with open("error.html","w") as f:
            f.write(html)
        raise NoLinksFoundException(summary_link)

    def get_last_body_href_databreaches(self,html): # this only works for databreaches.net
        soup = BeautifulSoup(html, 'html.parser')
        #get the div with the class "entry-content"
        entry_content=soup.find('div',attrs={'class':'entry-content'})
        #remove the div with class "crp_related" in the div
        links = entry_content.select('a:not(div.crp_related a)')
        

        if len(links)>0:
            return links[-1]['href']
        return None
    
        #NOTE: this can probably throw some exception, but I'm not sure what it would be.

    def get_last_body_href_generic(self,html):
        soup = BeautifulSoup(html, 'html.parser')
        
        #get the outermost div within the body
        body=soup.find('body')
        #get the last link in the body
        links = body.select('a')
        
        logging.debug("LINKS:"+str(links))

        if len(links)>0:
            last_link_index=len(links)-1
            last_link=links[last_link_index]
            while last_link.has_attr('href')==False or  not last_link['href'].startswith('http'): #this can be reworked later...
                last_link=links[last_link_index]
                last_link_index-=1

                if last_link_index<0:
                    break
        
        if last_link_index<0:
            return None
        else:
            return last_link['href']

    def get_read_more_href(self,html):
        pattern=r'Read more at.*?href="(.*?)"'
        href=re.findall(pattern,html,re.DOTALL)
        if len(href)==0:
            return None
        else:
            return href[0]
        
    
        
##-----------------Link Extractions-----------------##

##-----------------Text Preprocessing-----------------##
    def preprocess_html(self,html):
        for callback in self.preprocess_html_callbacks:
            
            html=callback(html)
        return html
    
    def trim_text(self,text):
        #return first 80% of text
        return text[:int(len(text)*0.8)]

    def get_article_link(self,article):
        return article['link']

    def remove_excess_whitespace(self,text):
        text=re.sub(r'\n+',' ',text)
        text=re.sub(r'\s+',' ',text)
        text=re.sub(r'\t+',' ',text)
        return text

    def remove_header(self,text):
        text=re.sub(r'<header.*?</header>','',text,flags=re.DOTALL)
        return text

    def remove_footer(self,text):
        text=re.sub(r'<footer.*?</footer>','',text,flags=re.DOTALL)
        return text

    def remove_navbar(self,text):
        text=re.sub(r'<nav.*?</nav>','',text,flags=re.DOTALL)
        return text

    def remove_ads(self,text):
        text=re.sub(r'<ins.*?</ins>','',text,flags=re.DOTALL)
        return text

    def remove_scripts(self,text):
        text=re.sub(r'<script.*?</script>','',text,flags=re.DOTALL)
        return text

    def remove_styles(self,text):
        text=re.sub(r'<style.*?</style>','',text,flags=re.DOTALL)
        return text
    
    def remove_head(self,text):  
        text=re.sub(r'<head.*?</head>','',text,flags=re.DOTALL)
        return text
    
    def remove_meta(self,text):
        text=re.sub(r'<meta.*?>','',text,flags=re.DOTALL)
        return text
    


    def extract_text(self,html):
        # 
        soup = BeautifulSoup(html.text, 'html.parser')
        full_text=soup.get_text()
        text=self.remove_excess_whitespace(full_text)
        #NOTE: DO OTHER HTML PREPROCESSING HERE
        return text

##-----------------Text Preprocessing-----------------##


##-----------------Utilities-----------------##
    def get_num_articles(self,headlines):
        return len(headlines)
    
    def add_reference(self,query_result,reference):
        query_result['Reference']=reference
        return query_result
    
    def add_score(self,query_result,score):
        query_result['Score']=str(score)+"/"+str(self.total_score)
        return query_result
    
    def add_region(self,query_result,region):
        query_result['Region']=region
        return query_result
    
    
    def make_request_with_session(self,url):
        try:
            session=requests.Session()

            response=session.get(url,headers=self.EXTERNAL_HEADERS,timeout=10)
            
        except requests.exceptions.MissingSchema as e:
            logging.error("Missing schema:"+str(e))
            return None
        except requests.exceptions.ReadTimeout as e:
            logging.error("Read timeout:"+str(e))
            return None
        # 
        return response

    def invoke_selenium(self,url):
        dr=webdriver.Safari()
        dr.get(url)
        bs=BeautifulSoup(dr.page_source,'html.parser')
        html=bs.prettify()
        return html
    
    def generate_mrkdwn(self,query_result):
        for field in query_result.keys():
            try:
                if type(query_result[field])==list or type(query_result[field])==set:#look at all these 'set's tomorrwo!!!
                    query_result[field]=', '.join(query_result[field])
            except TypeError as e:
                logging.error("TypeError in generate_mrkdwn:"+str(e))

        # return None <-- I forgot why I put this here, but I'm sure there was a reason...
            
        
        mrkdwn=self.mrkdwn_template.format(title=query_result['Title'],\
                                        organization=query_result['Victim Organization'],\
                                        location=query_result['Victim Location'],\
                                        sector=query_result['Sectors'],\
                                        threat_actor=query_result['Threat Actor'],\
                                        threat_actor_aliases=query_result["Threat Actor Aliases"],\
                                        malware=query_result['Malware'],\
                                        cves=query_result['CVEs'],\
                                        impact=query_result['Impact'],\
                                        key_judgement=query_result['Key Judgement'],\
                                        change_analysis=query_result['Change Analysis'],\
                                        timeline_of_activity=query_result['Timeline of Activity'],\
                                        summary=query_result['Summary'],\
                                        actor_motivation=query_result['Actor Motivation'],\
                                        reference=query_result['Reference'],\
                                        # score=query_result['Score'],\
                                            )
        
        return mrkdwn

    def message_zapier(self,mrkdwn,associated_webhook):
        if self.test_mode and not self.test_zapier:
            print(mrkdwn)
            return
        #make the post request, encode mrkdwn as utf-8
        logging.debug("Sending message to zapier @ {}.".format(associated_webhook))
        response=requests.post(associated_webhook,data=mrkdwn.encode('utf-8'))
        return response



    def label_article_category(self,headlines,category): #NOTE: this is a workaround for the fact that the API does not return the category of the article rn.
        for headline in headlines:
            headline['category']=category
        return headlines
    
    def map_country_to_region(self,country):
        try:
            return self.country_region_mapping[country]
        except KeyError:
            return "Other"
        except TypeError:
            if type(country)==list:
                return list(set([self.map_country_to_region(c) for c in country]))
    
    def map_region_to_webhook(self,region):
        try:
            if type(region)==set or type(region)==list:#NOTE:look back here later!!!
                return [self.region_webhook_mapping[r] for r in region]
            return [self.region_webhook_mapping[region]]
        
        except KeyError:
            logging.error("Region not found in mapping:"+str(region)+" with error:"+str(traceback.format_exc()))
            return self.region_webhook_mapping["Other"]
        except TypeError:
            logging.error("Region not found in mapping:"+str(region)+" with error:"+str(traceback.format_exc()))
            if type(region)==list: #technically this should never happen, but just in case...
                logging.info("Region is a list, mapping each element in list:{}".format(str(region)))
                return [self.map_region_to_webhook(r) for r in region]


    def check_if_na(self,field) -> bool:
        if field=="N/A" or field=="" or (type(field)==list and (len(field)==0 or field[0]=="N/A")):
            return True
        return False


##-----------------Utilities-----------------##




##-----------------Logging-----------------##
    def write_anomaly(self,article_id,error):
        self.anomalies_file.write(str(datetime.now())+"\n")
        self.anomalies_file.write(str(article_id)+"\n")
        self.anomalies_file.write(str(error)+"\n")
        self.anomalies_file.write("--------------------\n")
        self.mark_as_read(article_id)
        ##print traceback of error
        
        
        # exit()
        self.anomalies_file.flush()

##-----------------Logging-----------------##

    
##-----------------MAIN------------------##        

    def process_unread(self,article_id,article_category):

        article=self.get_article(article_id)
        if article_category==self.SUMMARY_CONTENT:
            article_link=self.get_article_link(article)#NOTE: change function name to get_article_link_summary later...  
            original_link=self.extract_article_link_from_summary(article_link)
        else:
            original_link=self.get_article_link(article)

        
        html=self.make_request_with_session(original_link) # change this to response ( left side)
        
        if html is None:
            raise NoHTMLFoundException(article_id)
        text=self.extract_text(html)

        try:
            query_result=self.gpt_query(text)
            if query_result is None:
                raise NoGPTResponseException(article_id)

        except openai.error.InvalidRequestError:

            while True:
                text=self.trim_text(text)
                try:
                    
                    query_result=self.gpt_query(text)
                    if query_result is None: 

                        return # to not mark as read NEW
                    self.mark_as_read(article_id)

                    break
                except openai.error.InvalidRequestError as e:

                    
                    continue

        query_result=self.add_reference(query_result,original_link)
        return query_result

    def fetch_and_label_headlines(self):
        headlines_full_content=self.get_headlines(category=self.FULL_CONTENT)
        headlines_summary_content=self.get_headlines(category=self.SUMMARY_CONTENT)

        headlines_full_content=self.label_article_category(headlines_full_content,self.FULL_CONTENT)
        headlines_summary_content=self.label_article_category(headlines_summary_content,self.SUMMARY_CONTENT)

        headlines=headlines_full_content+headlines_summary_content
        return headlines

    def write_test_results(self,query_result):

        with open("test_results/single_result_{}.json".format(datetime.now()),"w") as f:
                json.dump(query_result,f)


    def job(self):

        logging.debug("Starting job at "+str(datetime.now()))
        self.anomalies_file.write("Starting job at "+str(datetime.now())+"\n")
        self.session_id=self.login()
        unread_body=self.UNREAD_BODY
        unread_body['sid']=self.session_id

        headlines=self.fetch_and_label_headlines()
        
        query_results=[] #DELETE THIS LATER

        batch=[]
        test_count=0

        for headline in headlines:
            try:
                if self.test_mode and test_count==self.test_size: 
                    break
                
                try:
                    query_result=self.process_unread(headline['id'],headline['category'])
                    if self.test_mode:
                        query_results.append(query_result)#DELETE THIS LATER
                        test_count+=1
                        logging.info("Finished {} out of {} tests.".format(test_count,self.test_size))
                        logging.debug("IMDAT")



                    
                    result_score=self.score_gpt_response(query_result)
                    query_result=self.add_score(query_result,result_score)
                    if result_score<self.scoring_metric['threshold']: #NOTE: remove the false later
                        self.write_anomaly(headline['id'],"Score is below threshold at "+str(result_score))
                        self.mark_as_read(headline['id'])
                        logging.info("Score is below threshold at "+str(result_score))
                        self.write_test_results(query_result)

                        continue

                    #Skip the Report if there is no actor, malware or vulnerability 
                    if(self.check_if_na(query_result["Threat Actor"]) and self.check_if_na(query_result["Malware"]) and self.check_if_na(query_result["CVEs"])):
                        self.write_anomaly(headline['id'],"No actor, malware or vulnerability found")
                        self.mark_as_read(headline['id'])
                        logging.info("No actor, malware or vulnerability found")
                        self.write_test_results(query_result)

                        continue

                except Exception as e:
                    #HERE : in the future we should have a function that takes in the exception and does the hanndling.
                    logging.error(e)
                    self.mark_as_read(headline['id'])
                    self.write_anomaly(headline['id'],e)
                    continue
                
                if query_result is None: #way better idea is to wrap the whole thing in a try except for SyntaxError.
                    logging.error("Query result is None for article:"+str(headline['id']))
                    self.mark_as_read(headline['id'])
                    self.write_anomaly(headline['id'],"Query result is None")
                    continue

                self.mark_as_read(headline['id'])

                # result_score=self.score_gpt_response(query_result)
                # query_result=self.add_score(query_result,result_score)
                logging.debug(str(query_result)+'\n')
                logging.debug("Score:"+str(result_score))
                assigned_regions=self.map_country_to_region(query_result['Victim Location'])
                query_result=self.add_region(query_result,assigned_regions)

                corresponding_channels=self.map_region_to_webhook(assigned_regions)
                logging.debug("Corresponding channels:"+str(corresponding_channels))
                for channel in corresponding_channels:#buraya o message zapier parametresini de ekle
                    try:
                        markdown=self.generate_mrkdwn(query_result)
                        self.message_zapier(markdown,channel)
                    except Exception as e:
                        logging.error("Error during generating markdown or messaging zapier:"+str(e))
                        logging.error(traceback.format_exc())
                        self.mark_as_read(headline['id'])
                        # self.write_anomaly(headline['id'],e)
                        continue

                


                markdown=self.generate_mrkdwn(query_result)
                if self.test_mode:
                    logging.info("TEST MODE: writing test results.") # BURASI BURASI BURASI
                    self.write_test_results(query_result)

                batch.append(markdown)
                

            except NoLinksFoundException as e:
                logging.warning("[{}]No links found for article:".format(str(datetime.now()))+"<"+str(headline['id'])+">")
                self.mark_as_read(headline['id'])
                self.write_anomaly(headline['id'],e)

                continue # TODO: remove this

            except Exception as e:
                logging.error("[{}]Error in job:".format(str(datetime.now())))
                logging.error(traceback.format_exc())
                self.mark_as_read(headline['id'])
                self.write_anomaly(headline['id'],e)
                continue # <-- this doesnt really do anything unless i add other stuff in the loop
        


        #concatenate batch with newlines
        batch_concat='\n'.join(batch)
        
        
        # if self.batch:
        #     self.message_zapier(batch_concat)

        if self.test_mode:
            with open("test_results/bulk_result_{}.json".format(datetime.now()),"w") as f: ##COMMENT this later
                json.dump(query_results,f)
            with open("test_results/bulk_markdown_result_{}.txt".format(datetime.now()),"w") as f:
                f.write(batch)



def schedule_job(config,batch_mode=False):
    ttrss=TTRSS(config)
    if batch_mode:
        logging.info("Scheduling job every day at the following times:{}".format(str(config['message_times'])))
        for message_time in config['message_times']:
            schedule.every().day.at(message_time).do(ttrss.job)
    else:
        logging.info("Scheduling job every minute.")
        schedule.every().minute.do(ttrss.job)


    while True:
        try:
            schedule.run_pending()
            #sleep for 11 hours and 59 minutes
            if batch_mode:
                time.sleep(43140)
            else:
            #sleep for 59 seconds
                time.sleep(59)
        except Exception as e:
            logging.error("[{}]Error in schedule_job:".format(str(datetime.now())))
            logging.error(traceback.format_exc())
            continue

##-----------------MAIN------------------##

def clear_log(log_file):
    open(log_file,'w').close()

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', dest='config', default='config.json',
                        help='config file path', required=True)
    
    parser.add_argument('--test', dest='test',
                        help='Specify whether to run in test mode (no scheduling)',required=True)
    
    parser.add_argument('--test_size', dest='test_size', default=-1,
                        help='Specify the number of articles to test on.',required=False)

    parser.add_argument('--test_zapier', dest='test_zapier', default=None,
                        help='Specify whether to send messages to zapier in test mode.',required=False)
    
    parser.add_argument('--batch', dest='batch',
                        help='Specify whether to run in batch mode (send all articles at given intervals.)',required=False)
    
    parser.add_argument('--log_file',dest='log_file',help='Specify the log file path.',required=True)
    
    parser.add_argument('--clearlogs',dest='clear',help='Specify whether to clear the log file before starting.',required=False,default='false')

    parser.add_argument('--test_use_cheap_model',dest='test_use_cheap_model',help='Specify whether to use the cheap model for testing purposes.',required=False,default='false')
    
    parser.add_argument('--test_mark_as_read',dest='test_mark_as_read',help='Specify whether to mark articles as read in test mode.',required=False,default='true')
    

    args = parser.parse_args()
    logging.basicConfig(filename=args.log_file,level=logging.DEBUG,format='[%(levelname)s]@[%(asctime)s]: %(message)s')
    logging.info("Starting TTRSS with the following parameters:{}\n".format(str(args))+"\n")


    with open(args.config) as f:
        config=json.load(f)

    if args.clear=="true":
        clear_log(args.log_file)
    elif args.clear=="false":
        pass
    else:
        print("Invalid argument for --clearlogs, must be true or false")
        exit()

    print('hello')
    try:
        if args.test=="true":
            if args.test_size==-1:
                print("Please specify the number of articles to test on.")
                exit()
            if args.test_zapier==None:
                print("Please specify whether to send messages to zapier in test mode.")
                exit()
            config['test_mode']=True
            config['test']={}
            config['test']['test_size']=int(args.test_size)
            config['test']['test_zapier']=True if args.test_zapier=="true" else False
            config['test']['test_use_cheap_model']=True if args.test_use_cheap_model=="true" else False
            config['test']['test_mark_as_read']=True if args.test_mark_as_read=="true" else False
            # config['test_size']=int(args.test_size)
            # config['test_zapier']=True if args.test_zapier=="true" else False
            # config['test_use_cheap_model']=True if args.test_use_cheap_model=="true" else False

            ttrss=TTRSS(config)
            ttrss.job()
        elif args.test=="false":
            print("Starting TTRSS in production mode.")
            config['test_mode']=False
            schedule_job(config,True if args.batch=="true" else False)
        else:
            print("Invalid argument for --test, must be true or false")
    except Exception as e:
        logging.error("[{}]Error in main:".format(str(datetime.now())))
        logging.error(traceback.format_exc())



#TODO: check response status code from ttrss
#TODO: setup logger w/ datetime
#TODO: rework the way we handle SyntaxError exception caused by GPT response, see line 266
#NOTE: is there a limit on the size of the text we can send to slack?
#NOTE: i think gpt just outputs 'Not specified.' as the whole text if it can't find anything useful, do we
#catch the exception and just skip the article, or do we change the prompt?
#NOTE: sometimes the request takes too long, modify the default timeout for requests.
#NOTE: sometimes when the page has a banner, we get stuck.
#NOTE: sometimes the page has a paywall, we get stuck.
#NOTE: we need to add a timeout to gpt query.
#NOTE: 502 error from bad gateway when querying gpt, need to handle this.
#NOTE: we get SSL error from some sites, need to handle this.
#NOTE: better idea: DO THE MARK AS READ AFTER THE CALL TO THE PROCESS UNREAD FUNCTION, embed some logic there.
#NOTE: We are getting 403 errors from some sites, including cisa.gov
#NOTE: some urls are pdfs, we need to take this into account.
#NOTE: picus library integration with mwalwares ISSUE
#NOTE: slack mention region ISSUE
#NOTE: infer location,industry
#NOTE: added mark as read in write_anomaly, no need to call that whenever write_anomaly is called already.
#NOTE: change anomalies to JSON format
#NOTE: move errors to a separate file

#NOTE: id : 1 --> FullContent || id : 2 --> Summary Content            
    

##NOTE: I moved the following to FullContent:
#bleepingcomputer.com
#helpnetsecurity.com
#theregister.com
#thehackernews.com
#securityweek.com
#hackread.com
#thecyberwire.com
#threatpost.com


#ideas for refactoring:
#1. move TTRSS API calls to a separate class
#2. move GPT calls to a separate class
#3. move text preprocessing & link extraction to a separate class


#TODO: improve the exception handling mechanism to message me on Slack when an unkown error occurs.
#TODO: We have the problem of zapier returning 200 OK even when the message is not sent. We need to handle this.


