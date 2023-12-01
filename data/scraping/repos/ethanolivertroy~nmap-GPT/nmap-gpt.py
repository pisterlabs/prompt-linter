import argparse
import nmap
import openai

# Set OpenAI API key
openai.api_key = "your_openai_api_key"

# Define the command line arguments
parser = argparse.ArgumentParser(description='Nmap Network Scanner with OpenAI Analysis')
parser.add_argument('host', type=str, help='Host or IP address to scan')
parser.add_argument('-p', '--port', type=str, help='Port or port range to scan')

# Parse the command line arguments
args = parser.parse_args()

# Initialize Nmap
nm = nmap.PortScanner()

# Perform the Nmap scan
nm.scan(args.host, args.port)

# Extract the results of the Nmap scan
for host in nm.all_hosts():
    print('Host: {} ({})'.format(host, nm[host].hostname()))
    print('State: {}'.format(nm[host].state()))
    for protocol in nm[host].all_protocols():
        print('Protocol: {}'.format(protocol))
        lport = nm[host][protocol].keys()
        lport = sorted(lport)
        for port in lport:
            print('port: {}\tstate: {}'.format(port, nm[host][protocol][port]['state']))

            # Send a prompt to OpenAI for each open port
            if nm[host][protocol][port]['state'] == 'open':
                prompt = f"What security concerns should I be aware of given the {protocol} service on port {port} for host: {host}?"
                response = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=prompt,
                    max_tokens=1024,
                    n=1,
                    stop=None,
                    temperature=0.5,
                )

                # Extract the generated report from the OpenAI API response
                text = response["choices"][0]["text"]
                print("OpenAI Report:\n{}".format(text))
