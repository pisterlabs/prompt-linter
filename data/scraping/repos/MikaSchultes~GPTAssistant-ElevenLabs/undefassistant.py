import os
import shutil
import openai
import config
from pathlib import Path

assistantname = Path(__file__).stem
threadfilename = assistantname + "_thread.py"

src_dir = os.getcwd()
dest_dir = src_dir +  "\\" + threadfilename
shutil.copy('undefthread.py', dest_dir)

openai.api_key = config.openaiapikey
thread = openai.beta.threads.create()

configfile= assistantname + "_thread_conf.py"
with open(configfile, "w") as file:
    l1 = "threadid = "
    l2 = '"'
    l3 = thread.id
    l4 = '"'
    file.writelines([l1, l2, l3, l4])
    file.close

print(f"A thread for {assistantname} has been created. you can Talk to {assistantname} by executing {threadfilename}")