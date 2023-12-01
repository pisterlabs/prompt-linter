"""Question 1 & 2 Prompts """
import anthropic
import json, sys
import sqlite3
import datetime, time
import pandas as pd
from bs4 import BeautifulSoup
import pprint
pp = pprint.PrettyPrinter(indent=4)

dbfolder = './db/'
EarningsTranscriptList = './db/EarningsTranscriptList.csv'
EarningsTranscriptDB = './db/EarningsTranscript.db'
q1_DB = './db/question1.json'
q3_DB = './db/question3.json'
conn = sqlite3.connect(EarningsTranscriptDB)

#### Anthropic ####
claude_key = json.load(open('../claude_key.json'))
client = anthropic.Client(api_key=claude_key)
human = json.load(open('./static/prompts.json'))

def get_document_count():
    # Call the get_document_count function to retrieve the count
    cursor = conn.execute("SELECT COUNT(*) FROM files")
    result = cursor.fetchone()
    if result:
        document_count = result[0]
        return document_count
    else:
        return 0
    
document_count = get_document_count()
print("Number of documents:", document_count)

query = f"SELECT * FROM files"  # Customize the SQL query based on the selected columns
df_db = pd.read_sql_query(query, conn)
Q1_DB_hist = json.load(open(q1_DB, 'r'))

def count_tokens(string: str, text=''):
    num_tokens = anthropic.count_tokens(string)
    print(f"{text} number of tokens: {num_tokens}")
    return num_tokens

def promptInstance(files, dateStr, qtrSelect):
    files['date'] = pd.to_datetime(files['date'])
    cur_file = files[files['date']==pd.Timestamp(dateStr)].squeeze()
    print('cur_file',cur_file['date'])
    # dateSeries =  pd.to_datetime(files['date'])
    prevQtrDate = sorted(files['date'])[-2]
    prevQtr = str(prevQtrDate.date()).replace('-','')

    if qtrSelect=='2':
        # Get previous quarter end date
        prev_file = files[files['date']==prevQtrDate].squeeze()
        fileID_prev = cur_file['Ticker']+prevQtr
        print('prev_file',prev_file['date'])
        prev_textStr = prev_file['content']
        # Start Strings
        prev_start_idx = max(prev_textStr.find('Stock Advisor returns as of'),prev_textStr.find('See the 10 stocks'),prev_textStr.find('Prepared Remarks'))
        count_tokens(prev_textStr[prev_start_idx:], 'Previous Qtr')
    else:
        fileID_prev = ''

    company = cur_file['company']
    qtr = cur_file['qtr']
    year = cur_file['year']
    cur_textStr = cur_file['content']
    fileID = cur_file['Ticker']+dateStr

    # Start Strings
    cur_start_idx = max(cur_textStr.find('Stock Advisor returns as of'),cur_textStr.find('See the 10 stocks'),cur_textStr.find('Prepared Remarks'))
    
    # Combining 2 questions - sunsetting this
    # prompt=f"{anthropic.HUMAN_PROMPT} {human['q1a']}{company} for period {fileDateStr} : <current>{cur_textStr[cur_start_idx:]}</current> \
    #         \n\n {human['q1b']} <previous> {prev_textStr[prev_start_idx:] } </previous>. \n\n{human['s1a']} {human['q1c']} {human['q1c2']} \
    #         {human['q1d']} {human['q1e']} {human['q1f']} {human['q1g']} {human['q1h']} {human['q2a']} {human['q2b']} {human['q2c']} {anthropic.AI_PROMPT}"
    
    # Question 1current
    prompt1c=f"{anthropic.HUMAN_PROMPT} {human['q1a']}{company} for period ended {qtr} {year} : <current>{cur_textStr[cur_start_idx:]}</current> \
            \n\n {human['s1b']} \n\n{human['q1c']} {human['q1c2']} {human['q1d']} {human['q1e']} {human['q1f']} {human['q1g']} {human['q1h']} {human['q1h2']}\
            {anthropic.AI_PROMPT}"
    if qtrSelect=='2':
        # Question 1previous
        prompt1p=f"{anthropic.HUMAN_PROMPT} {human['q1b']}{company} : <previous>{prev_textStr[prev_start_idx:]}</previous> \
                \n\n {human['s1b']} \n\n{human['q1c']} {human['q1c2']} {human['q1d']} {human['q1e']} {human['q1f']} {human['q1g']} {human['q1h']} {human['q1h2']}\
                {anthropic.AI_PROMPT}"    
    else: prompt1p=''
    # Question 2 - sunsetting this
    # prompt2=f"{anthropic.HUMAN_PROMPT} {human['s2a']} <previous> {prev_textStr[prev_start_idx:] } </previous>. {human['s2b']} {human['q2a']} \
    #         {human['q2b']} {human['q2c']}  {anthropic.AI_PROMPT}"
    
    # Question 2combined 
    prompt2c=f"{anthropic.HUMAN_PROMPT} {human['s2a2']}{company}. {human['s2b']} {human['q2a']} {human['q2b']} {human['q2c']}  {anthropic.AI_PROMPT}"        
        
    count_tokens(cur_textStr[cur_start_idx:], 'Current Qtr')
    num_tokens = count_tokens(prompt1c+prompt1p+prompt2c, 'prompt Total')
    
    if num_tokens>=100000:
        print('Total number of tokens exceeded the model limit!\nStopping the process now.')
        sys.exit()
        
    return fileID, fileID_prev, prompt1c, prompt1p, prompt2c


def ClaudeAPI(prompt, max_tokens=10000, model="claude-v1.3-100k"):
    # Calling API
    resp = client.completion(
        prompt = prompt,
        stop_sequences = [anthropic.HUMAN_PROMPT],
        model = model,
        max_tokens_to_sample = max_tokens,
    )
    return resp
    
def response_to_db(response1,response2, fileID):
    # response1 - question1
    # response2 - question2
    tags = ['financials','catalyst','qa','rating','overall','title','compfin','change','rate2','reasons']
    
    html = f"""{response1} {response2}"""
    soup = BeautifulSoup(html, 'html.parser')

    tagsDict={}
    for t in tags:
        if soup.find(t) is not None:
            tagsDict[t] = soup.find(t).text.strip()
        
    if fileID not in Q1_DB_hist.keys():
        Q1_DB_hist[fileID] = tagsDict
    else:
        addPairs = [x for x in tagsDict.keys() if x not in Q1_DB_hist[fileID].keys()]
        for k in addPairs:
            Q1_DB_hist[fileID][k] = tagsDict[k]

    pp.pprint(Q1_DB_hist[fileID])
    ### Save to json db
    with open(q1_DB, 'w') as f:
        json.dump(Q1_DB_hist, f)
        print('Records save to: ', q1_DB)

    return Q1_DB_hist[fileID]

######## INPUT ###########
def input_module():
    ticker = input('Enter the Ticker: ').upper()
    qtrSelect = input('Enter <1> or <2> quarters (2:current & previous) to run, <Enter> for default 2: ')
    if qtrSelect=='':
        qtrSelect='2'
    dateStr = input('Enter the requested quarter end date in <YYYYmmdd> format, <Enter> for default 20230331: ')
    if dateStr=='':
        dateStr='20230331'
    print('Available Models: "claude-v1.3-100k", "claude-v1-100k", "claude-instant-v1-100k"')
    modelSelect = "claude-v1.3-100k"

    return ticker,dateStr,qtrSelect,modelSelect

def main(ticker,dateStr,qtrSelect,modelSelect):
    files = df_db[(df_db['Ticker']==ticker)]
    print(files)



    fileID,fileID_prev,prompt1c,prompt1p,prompt2c = promptInstance(files,dateStr,qtrSelect)

    print(f'Running on model: {modelSelect}')
    # Run the stated quarter first
    start_time = time.time()
    response = ClaudeAPI(prompt1c, 10000, modelSelect)
    resp_1c = response['completion']
    time1 = time.time()
    print(f'Elapsed time for Claude: {(time1-start_time):.2f} seconds')
    # pp.pprint(resp_1c)
    if response['stop_reason']!='stop_sequence':
        print(response['stop_reason'])
        print('Error in Claude API, exiting...')
        sys.exit()

    if qtrSelect=='1' or qtrSelect==1:
        resp_ques2=''
        resp2json = response_to_db(resp_1c,resp_ques2, fileID)
        print(f'Claude response saved for: {fileID}')

    elif qtrSelect=='2' or qtrSelect==2:
        # Check if Claude had already analysed previous quarter 
        if fileID_prev in Q1_DB_hist.keys():
            resp_1p = Q1_DB_hist[fileID_prev]
            print('Previous qtr output from Claude already existed, proceed to compare')
        else:
            # No previous saved Claude response
            print('No previous Claude input - feeding previous Qtr transcript')
            start_time = time.time()
            response = ClaudeAPI(prompt1p, 10000, modelSelect)
            resp_1p = response['completion']
            time1 = time.time()
            print(f'Elapsed time for Claude: {(time1-start_time):.2f} seconds')
            # pp.pprint(resp_1p)
            if response['stop_reason']!='stop_sequence':
                print(response['stop_reason'])
                print('Error in Claude API, exiting...')
                sys.exit()
            # Save previous quarter
            resp_ques2=''
            resp2json = response_to_db(resp_1p,resp_ques2, fileID_prev)
            print(f'Claude response saved for: {fileID_prev}')

        # Taking an API break
        time.sleep(10)
        # Combining previous and current to compare
        print('Feeding consolidated prompt for comparison question ...')
        comboPrompt = prompt1c+resp_1c+prompt1p+resp_1p+prompt2c
        count_tokens(comboPrompt)
        start_time = time.time()
        response = ClaudeAPI(comboPrompt, 10000, modelSelect)
        resp_combo = response['completion']
        time1 = time.time()
        print(f'Elapsed time for Claude: {(time1-start_time):.2f} seconds')
        # pp.pprint(resp_combo)
        if response['stop_reason']!='stop_sequence':
            print(response['stop_reason'])
            print('Error in Claude API, exiting...')
            sys.exit()
        resp2json = response_to_db(resp_1c,resp_combo, fileID)

        print(f'Completed for {ticker} on the 2 questions')
        
        
##sid##
#for the case of peer comps, only 2 company's earnings to be compared
def promptInstance2(file,file2):
    company1,company2 = file['company'],file2['company']
    qtr1,qtr2 = file['qtr'],file2['qtr']
    year1,year2 = file['year'],file2['year']
    textStr1,textStr2 = file['content'],file2['content']
    filedate = str(pd.to_datetime(file['date']).date()).replace('-','')
    fileID = file['company']+','+file2['company']+','+filedate

    # Start String
    start_idx1 = max(textStr1.find('Stock Advisor returns as of'),textStr1.find('See the 10 stocks'),textStr1.find('Prepared Remarks'))
    start_idx2 = max(textStr2.find('Stock Advisor returns as of'),textStr2.find('See the 10 stocks'),textStr2.find('Prepared Remarks'))
        
    prompt=f"{anthropic.HUMAN_PROMPT} {human['q3']}{human['q3a']}{company1},{company2}.\
    Earnings briefing for{company1} is in the <Transcript1> tag: <Transcript1>{textStr1[start_idx1:]}</Transcript1>\
    Earnings briefing for{company2} is in the <Transcript2> tag: <Transcript2>{textStr2[start_idx2:]}</Transcript2>\
    \n{human['q3b']}. {human['q3c']} {human['q3d']} {human['q3e']}\
    {human['q3f']}{human['q3g']}{human['q3h']}    \
    {anthropic.AI_PROMPT}"

    num_tokens = count_tokens(prompt)    
    if num_tokens>=100000:
        print('Total number of tokens exceeded the model limit!\nStopping the process now.')
        sys.exit()
        
    return fileID, prompt

##sid##


##sid##
#Creating prompt to be fed to Claude, for the instance of 3 company's earnings to be compared
#returns an id, and the generated prompt 
def promptInstance3(file,file2,file3):
    company1,company2,company3 = file['company'],file2['company'],file3['company']
    qtr1,qtr2,qtr3 = file['qtr'],file2['qtr'],file3['qtr']
    year1,year2,year3 = file['year'],file2['year'],file3['year']
    textStr1,textStr2,textStr3 = file['content'],file2['content'],file3['content']
    filedate = str(pd.to_datetime(file['date']).date()).replace('-','')
    fileID = file['company']+','+file2['company']+','+file3['company']+','+filedate

    # Start String
    start_idx1 = max(textStr1.find('Stock Advisor returns as of'),textStr1.find('See the 10 stocks'),textStr1.find('Prepared Remarks'))
    start_idx2 = max(textStr2.find('Stock Advisor returns as of'),textStr2.find('See the 10 stocks'),textStr2.find('Prepared Remarks'))
    start_idx3 = max(textStr3.find('Stock Advisor returns as of'),textStr3.find('See the 10 stocks'),textStr3.find('Prepared Remarks'))

        
    #TO add year and period
    prompt=f"{anthropic.HUMAN_PROMPT} {human['q3']}{human['q3a']}{company1},{company2},{company3}.\
    Earnings briefing for{company1} is in the <Transcript1> tag: <Transcript1>{textStr1[start_idx1:]}</Transcript1>\
    Earnings briefing for{company2} is in the <Transcript2> tag: <Transcript2>{textStr2[start_idx2:]}</Transcript2>\
    Earnings briefing for{company3} is in the <Transcript3> tag: <Transcript3>{textStr3[start_idx3:]}</Transcript3>\
    \n{human['q3b']}. {human['q3c']} {human['q3d']} {human['q3e']}\
    {human['q3f']}{human['q3g']}{human['q3h']}    \
    {anthropic.AI_PROMPT}"

    num_tokens = count_tokens(prompt)    
    if num_tokens>=100000:
        print('Total number of tokens exceeded the model limit!\nStopping the process now.')
        sys.exit()
        
    return fileID, prompt

##sid##


##sid##
#function organising and storing response from claude into a json file for case of peer comps
def response_to_db2(response, fileID):
    Q3_DB_hist = json.load(open(q3_DB, 'r'))

    html = f"""{response}"""
    soup = BeautifulSoup(html, 'html.parser')

    financials = soup.find('financials').text.strip()
    challenges = soup.find('challenges').text.strip()
    growth = soup.find('growth').text.strip()
    resilience = soup.find('resilience').text.strip()
    leadership = soup.find('leadership').text.strip()
    insider = soup.find('insider').text.strip()
    ranking = soup.find('ranking').text.strip()
    

    Q3_DB_hist[fileID] = dict({'financials': financials,
                                'challenges': challenges,
                                'growth': growth,
                                'resilience': resilience,
                                'leadership': leadership,
                                'insider' : insider,
                                'ranking' : ranking
                                })

    with open(q3_DB, 'w') as f:
        json.dump(Q3_DB_hist, f)
        print('Records save to: ', q3_DB)

    return Q3_DB_hist[fileID]
##sid##

##sid##
#Input function for cross company comparison
def peer_input_funct():
    ticker = input('Enter the first Ticker: ').upper()
    ticker2 = input('Enter the second Ticker: ').upper()
    ticker3 = input('Enter the third Ticker: ').upper()
    dateStr = input('Enter the quarter end date in <YYYYmmdd> format, (e.g. 20230331) : ')
    return ticker,ticker2,ticker3,dateStr
##sid##

##sid##
def peer_comp(ticker,ticker2,ticker3,dateStr):
    print("Running peer comps function")
    fileDateStr = datetime.datetime.strftime(pd.Timestamp(dateStr), '%B %d, %Y')
    file = df_db[(df_db['Ticker']==ticker) & (df_db['date']==fileDateStr)].squeeze()
    file2 = df_db[(df_db['Ticker']==ticker2) & (df_db['date']==fileDateStr)].squeeze()
    file3 = df_db[(df_db['Ticker']==ticker3) & (df_db['date']==fileDateStr)].squeeze()
    #selecting between 2 company and 3 company comparison based on input
    if len(file)>0 and len(file2)>0 and len(file3)>0:
        fileID, prompt = promptInstance3(file,file2,file3)
    else:
        fileID, prompt = promptInstance2(file,file2)
    
    print('Available Models: "claude-v1.3-100k", "claude-v1-100k", "claude-instant-v1-100k"')
    print('Running on model: claude-v1.3-100k')
    resp3 = ClaudeAPI(prompt, 100000, "claude-v1.3-100k")
    resp2json = response_to_db2(resp3['completion'], fileID)
    print(f"Completed generating and storing peer comps Claude response for {ticker},{ticker2},{ticker3}.")
    
    #maybe we can use the returned fileID to fetch from the json?
    return fileID
##sid## 
        

if __name__=='__main__':
    ticker,dateStr,qtrSelect,modelSelect = input_module()
    main(ticker,dateStr,qtrSelect,modelSelect)
    #Optional runnning of peer comps, the boolean below can be replace with connection to a button click or something later
    peer_comps = False
    if peer_comps:
        ticker,ticker2,ticker3,dateStr = peer_input_funct()
        fileID = peer_comp(ticker,ticker2,ticker3,dateStr)
