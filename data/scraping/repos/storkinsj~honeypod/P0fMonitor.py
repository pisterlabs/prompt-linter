#!/usr/bin/python3
import subprocess
import openai
import os
import sys
import psutil
from socket import AF_INET


log_env_name = "HONEYPOD_LOG"
key_env_name = "OPENAI_KEY"

if "honeylog" in os.environ:
    honeylog = os.getenv(log_env_name)
else:
    honeylog = '/var/log/honeypod'

#
# Get my interface; default eth0
#
network_interface = "eth0"
ip_address = ""


net_interfaces = psutil.net_if_addrs()

found = False
for interface, addresses in net_interfaces.items():
    for address in addresses:                                            
        if interface != "lo" and not address.address.startswith("127."): 
            network_interface = interface                                
            ip_address = address.address
            found = True
            break
    if found:
        break

class P0fMonitor:
    def __init__(self, api_key):
        self.api_key = api_key

    def monitor_p0f(self, program_path, log_path):
            #open running listen on all ports ignoring outgoing requests from this host
        print (f"\'not src host {ip_address}\'")
        process = subprocess.Popen([program_path, "-i", network_interface, "-u", "p0f", f"\'not src host {ip_address}\'"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
#        process = subprocess.Popen([program_path, f"-i{network_interface}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        #output_file = open(log_path, 'a')
        start_token = ".-["
        end_token = "----"
        msg = ""
        res = ""
        for line in  process.stdout:
            print(line.strip())
            if line.startswith(start_token):
                msg = line
            elif end_token in line:
                if not 'mtu' in msg:
                   self.send_question(msg, log_path)  
            else:
                msg = msg + " " + line

        process.wait()
        #output_file.close()
        return process.returncode

    def send_question(self, p0f_output, path_of_log):
        openai.api_key = self.api_key
        question =  "analyze the signature and origin of this output from p0f based on destination port:"+p0f_output

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[

                {"role": "user", "content": question}
            ]
        )

        answer = response.choices[0].message.content
        print(answer)
        command = 'echo "HONEYPOD: Suspicious network traffic: {} {}" >> {}'.format(p0f_output, answer, path_of_log)
        os.system(command) 
   

# Example usage
def main():
    if key_env_name in os.environ:
        api_key = os.getenv(key_env_name)
    else:
        print("API KEY MISSING; ABORTING")
        sys.exit(1)
    monitor = P0fMonitor(api_key)
    p0f_program_path = "/app/p0f-master/p0f"
    monitor.monitor_p0f(p0f_program_path, honeylog)

if __name__ == '__main__':
    main()

