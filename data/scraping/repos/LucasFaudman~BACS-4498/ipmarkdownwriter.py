from analyzerbase import *
from markdownwriter.markdownwriterbase import *


from netanalyzers.ipanalyzer import IPAnalyzer
from loganalyzers.cowrieloganalyzer import CowrieLogAnalyzer, Attack
from openaianalyzers.openaianalyzer import OpenAIAnalyzer

from .visualizer import CounterGrapher


class IPAnalyzerMarkdownWriter(MarkdownWriter):


    def prepare(self):
    
        self.md += h1("What do you know about the attacker?")
        self.md_editors.append(self.add_ip_locations)
        self.md_editors.append(self.add_shodan)
        self.md_editors.append(self.add_isc)
        self.md_editors.append(self.add_threatfox)
        self.md_editors.append(self.add_cybergordon)
        self.md_editors.append(self.add_whois)
        



    def add_ip_locations(self, md, data):
        md += h2("IP Locations")
        location_data = [(ip, 
                        data[ip]["isc"]["ip"]["ascountry"],
                        data[ip]["isc"]["ip"]["as"],
                        data[ip]["isc"]["ip"]["asname"],
                        data[ip]["isc"]["ip"]["network"]
                        ) for ip in data if data[ip]["isc"].get("ip")]

        md += table(['IP Address', 'Country', "AS", "AS Name", "Network"], location_data)
        return md
    


    def add_isc(self, md, data):
        

        if len(data) == 1:
            sharing_url = list(data.values())[0]["isc"]["sharing_link"]
            sharing_link = link(sharing_url, sharing_url)
        else:
            sharing_link = link("https://isc.sans.edu/ipinfo/", "https://isc.sans.edu/ipinfo/")

        md += h2("Internet Storm Center (ISC) " + sharing_link)
        
        ics_data = [(ip, 
                        data[ip]["isc"]["ip"]["count"],
                        data[ip]["isc"]["ip"]["attacks"],
                        data[ip]["isc"]["ip"]["mindate"],
                        data[ip]["isc"]["ip"]["maxdate"],
                        data[ip]["isc"]["ip"]["updated"],
                        ) for ip in data if data[ip]["isc"].get("ip")]
        
        headers = ['IP Address', 'Total Reports', "Targets", "First Report", "Last Report", "Update Time"]
        md += table(headers, ics_data)
        return md    



    def add_whois(self, md, data):
        md += h2("Whois")

        #md += table(['IP Address', 'Whois Data'], whois_data)
        for ip in data:
            whois_data = data[ip]["whois"]

            #sharing_link = f'<a href="{whois_data["sharing_link"]}" >'
            sharing_link = link(whois_data["sharing_link"], whois_data["sharing_link"])
            #md += h3(f"Whois data for: {ip} " + sharing_link)
            whois_inner_md = h3(f"Whois data for: {ip} " + sharing_link)
            whois_codeblock = codeblock(whois_data["results"].get("whois_text", whois_data["error"]))
            md += collapseable_section(whois_inner_md + whois_codeblock,
                                       f"Whois data for: {ip}", #+ sharing_link,
                                       header_level=2,
                                       )
            # if isinstance(whois_data, str):
            #     md += codeblock(whois_data)
            # else:    
            #     md += codeblock(whois_data["whois_text"])

        return md



    def add_cybergordon(self, md, data):
        md += h2("CyberGordon")
        for ip in data:
            cybergordon_data = data[ip]["cybergordon"]
            
            if cybergordon_data.get("error"):
                #md += codeblock(cybergordon_data)
                continue
            
                
            sharing_link = link(cybergordon_data["sharing_link"], cybergordon_data["sharing_link"])
            
            md += h3(f"Cybergordon results for: {ip} " + sharing_link)
            cybergordon_table_data = []
            for priority in ["high", "medium", "low"]:
                for entry in cybergordon_data["results"].get(priority, []):
                    cybergordon_table_data.append((entry["engine"], entry["result"], entry["url"]))

            cybergordon_table_headers = ['Engine', 'Results', "Url"]
            md += table(cybergordon_table_headers, cybergordon_table_data)
        
        return md
    


    def add_shodan(self, md, data):
        md += h2("Shodan")
        for ip in data:
            
            shodan_data = data[ip]["shodan"]

            if shodan_data.get("error"):
                #md += codeblock(shodan_data)
                continue

            sharing_link = link(shodan_data["sharing_link"], shodan_data["sharing_link"])
            md += h3(f"Shodan results for: {ip} " + sharing_link)
            headers = list(shodan_data["results"]["general"].keys())
            shodan_general_values = list(shodan_data["results"]["general"].values())
            md += table(headers, [shodan_general_values])
            
            md += h4("Open Ports")
            shodan_ports_data = [(port, entry["protocol"], entry["service_name"], entry["timestamp"]) for port, entry in shodan_data["results"]["ports"].items()]
            shodan_ports_keys = ["Port", "Protocol", "Service", "Update Time"]
            md += table(shodan_ports_keys, shodan_ports_data)
        return md



    def add_threatfox(self, md, data):
        md += h2("Threat Fox")
        for ip in data:

            threat_fox_data = data[ip]["threatfox"]
            if threat_fox_data.get("error"):
                #md += codeblock(threat_fox_data)
                continue

            sharing_link = link(threat_fox_data["sharing_link"], threat_fox_data["sharing_link"])
            md += h3(f"Threat Fox results for: {ip} " + sharing_link)
            
            
            threatfox_ioc_table_values = []
            threatfox_malware_table_values = []

            for ioc in threat_fox_data["results"]:
                ioc_link = link(ioc["ioc"], ioc["ioc_url"])
                malware_link = link(ioc["malware"], ioc["malware_url"])

                ioc_data = ioc["ioc_data"]
                threatfox_ioc_table_values.append((ioc_link, ioc_data["Threat Type"], malware_link, ioc_data["Malware alias"], ioc_data["Confidence Level"], ioc_data["First seen"], ioc_data["Last seen"]))

                malware_data = ioc["malware_data"]
                threatfox_malware_table_values.append((malware_link, malware_data["Malware alias"], malware_data["Number of IOCs"], malware_data["First seen"], malware_data["Last seen"], malware_data["Malpedia"]))


            md += h4(f"ThreatFox IOCS (Indicators of Compromise) Found for {ip}")            
            threatfox_ioc_table_headers = ["IOC", "Threat Type", "Malware", "Malware Alias", "Confidence Level", "First Seen", "Last Seen"]            
            md += table(threatfox_ioc_table_headers, threatfox_ioc_table_values)

            md += h4(f"ThreatFox Malware Found for {ip}")
            threatfox_malware_table_headers = ["Malware", "Aliases", "Number of IOCs", "First Seen", "Last Seen", "Malpedia URL"]
            md += table(threatfox_malware_table_headers, set(threatfox_malware_table_values))

        
        return md


