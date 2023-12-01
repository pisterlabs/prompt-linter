import nmap
import openai
import ipaddress
import time
import warnings
import os
import pandas as pd
from io import StringIO 
from datetime import date

warnings.filterwarnings('ignore')

def load_banner():
    print()
    print('''\x1b[36m
      ___ ___                            
 |   Y   .-----.--------.---.-.-----.
 |.  1  /|     |        |  _  |  _  |
 |.  _  \|__|__|__|__|__|___._|   __|
 |:  |   \                    |__|   
 |::.| .  )                          
 `--- ---'    
 ------------------------------------
 Incase of any bug report in github
 author~@sudhanshu-patel
 ------------------------------------\x1b[37m''')
    
def knowledgeMAP():
    try:
        ipaddr = input('[!] Enter the target IP address: ')
        ip_object = ipaddress.ip_address(ipaddr)
    except ValueError:
        print(f'\x1b[31m[x] The IP address {ipaddr} is not valid')
        return

    target = str(ipaddr)
    print('\x1b[32m[*] Scanning in progress....\x1b[37m')
    start = time.time()
    nm = nmap.PortScanner()
    nm.scan(target)
    output = nm.csv()

    csvStringIO = StringIO(output)
    df = pd.read_csv(csvStringIO, sep=";", index_col=False)
    df.drop(['host','hostname','hostname_type'], inplace=True, axis=1)

    try:
        openai.api_key = input("[!] Enter your openai api key: ")
        print('\x1b[32m[*] Generating report....\x1b[37m')
        messages = [
            {"role": "system", "content": "You are a professional vulnerability scanner"},
        ]

        message = f'''Generate a profession vulnerability assessment report with this nmap output- 
        {df}. Include vulnerable CVEs for each port depending on the version, recommendation, plan of action, conclusion. Also don't include ip address, hostname and date in the report.'''
        if message:
            messages.append(
                {"role": "user", "content": message},
            )
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )

        reply = chat.choices[0].message.content
        end = time.time()
    except:
        print("\x1b[31m[x] Invalid API key")
        return
    
    print(f'\x1b[32m[(っ^з^)♪♬]Report generated successfully\x1b[37m')
    filename = input("\n[!] Enter filename for report: ")
    f = open(filename + '.txt',"w+")
    f.write(f'''
    Vulnerability Assessment Report
    ===============================

    Assessed Target
    ---------------
    IP Address: {target}
    Assessment Date: {str(date.today())}

    {reply}
    ''')
    f.close()
    curr_dir = os.getcwd()
    curr_path = os.path.join(curr_dir,filename+'.txt')
    print(f'\x1b[32m[(っ^з^)♪♬] The report is saved at: {curr_path} - total time taken: {(end-start)*10:.03f} seconds')


load_banner()
knowledgeMAP()


