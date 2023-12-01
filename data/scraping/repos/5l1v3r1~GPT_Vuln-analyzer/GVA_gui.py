import customtkinter
import openai
import nmap
import dns.resolver
from subprocess import run
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.title("GVA - GUI")
root.geometry("600x400")

nm = nmap.PortScanner()
model_engine = "text-davinci-003"


def application():
    try:
        apikey = entry1.get()
        openai.api_key = apikey
        target = entry2.get()
        attack = entry5.get()
        outputf = str(entry4.get())
        match attack:
            case 'geo':
                val = geoip(apikey, target)
                print(val)
                output_save(val, outputf)
            case "nmap":
                p = int(entry3.get())
                match p:
                    case 1:
                        val = p1(target)
                        print(val)
                        output_save(val, outputf)
                    case 2:
                        val = p2(target)
                        print(val)
                        output_save(val, outputf)
                    case 3:
                        val = p3(target)
                        print(val)
                        output_save(val, outputf)
                    case 4:
                        val = p4(target)
                        print(val)
                        output_save(val, outputf)
                    case 5:
                        val = p5(target)
                        print(val)
                        output_save(val, outputf)
            case "dns":
                val = dnsr(target)
                output_save(val, outputf)
            case "subd":
                val = sub(target)
                output_save(val, outputf)
    except KeyboardInterrupt:
        print("Keyboard Interrupt detected ...")


def geoip(key, target):
    url = "https://api.ipgeolocation.io/ipgeo?apiKey={a}&ip={b}".format(
        a=key, b=target)
    content = run("curl {}".format(url))
    return content


def output_save(output, outf):
    top = customtkinter.CTkToplevel(root)
    top.title("GVA Output")
    top.grid_rowconfigure(0, weight=1)
    top.grid_columnconfigure(0, weight=1)
    top.textbox = customtkinter.CTkTextbox(
        master=top, height=500, width=400, corner_radius=0)
    top.textbox.grid(row=0, column=0, sticky="nsew")

    try:
        file = open(outf, 'x')
    except FileExistsError:
        file = open(outf, "r+")
    file.write(str(output))
    file.close
    top.textbox.insert("0.0", text=output)


def sub(target):
    s_array = ['www', 'mail', 'ftp', 'localhost', 'webmail', 'smtp', 'hod', 'butterfly', 'ckp',
               'tele2', 'receiver', 'reality', 'panopto', 't7', 'thot', 'wien', 'uat-online', 'Footer']

    ss = []
    out = ""
    for subd in s_array:
        try:
            ip_value = dns.resolver.resolve(f'{subd}.{target}', 'A')
            if ip_value:
                ss.append(f'{subd}.{target}')
                if f"{subd}.{target}" in ss:
                    print(f'{subd}.{target} | Found')
                    out += f'{subd}.{target}'
                    out += "\n"
                    out += ""
                else:
                    pass
        except dns.resolver.NXDOMAIN:
            pass
        except dns.resolver.NoAnswer:
            pass
        except KeyboardInterrupt:
            print('Ended')
            quit()
    return out


def dnsr(target):
    analize = ''
    record_types = ['A', 'AAAA', 'NS', 'CNAME', 'MX', 'PTR', 'SOA', 'TXT']
    for records in record_types:
        try:
            answer = dns.resolver.resolve(target, records)
            for server in answer:
                st = server.to_text()
                analize += "\n"
                analize += records
                analize += " : "
                analize += st
        except dns.resolver.NoAnswer:
            print('No record Found')
            pass
        except KeyboardInterrupt:
            print("Bye")
            quit()
    try:
        prompt = "do a DNS analysis of {} and return proper clues for an attack in json".format(
            analize)
        # A structure for the request
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
        )
        response = completion.choices[0].text
    except KeyboardInterrupt:
        print("Bye")
        quit()
    return response


def p1(ip):
    nm.scan('{}'.format(ip), arguments='-Pn -sV -T4 -O -F')
    json_data = nm.analyse_nmap_xml_scan()
    analize = json_data["scan"]
    try:
        # Prompt about what the quary is all about
        prompt = "do a vulnerability analysis of {} and return a vulnerabilty report in json".format(
            analize)
        # A structure for the request
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1
        )
        response = completion.choices[0].text
    except KeyboardInterrupt:
        print("Bye")
        quit()
    return response


def p2(ip):
    nm.scan('{}'.format(ip), arguments='-Pn -T4 -A -v')
    json_data = nm.analyse_nmap_xml_scan()
    analize = json_data["scan"]
    try:
        # Prompt about what the quary is all about
        prompt = "do a vulnerability analysis of {} and return a vulnerabilty report in json".format(
            analize)
        # A structure for the request
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1
        )
        response = completion.choices[0].text
    except KeyboardInterrupt:
        print("Bye")
        quit()
    return response


def p3(ip):
    nm.scan('{}'.format(ip), arguments='-Pn -sS -sU -T4 -A -v')
    json_data = nm.analyse_nmap_xml_scan()
    analize = json_data["scan"]
    try:
        # Prompt about what the quary is all about
        prompt = "do a vulnerability analysis of {} and return a vulnerabilty report in json".format(
            analize)
        # A structure for the request
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1
        )
        response = completion.choices[0].text
    except KeyboardInterrupt:
        print("Bye")
        quit()
    return response


def p4(ip):
    nm.scan('{}'.format(ip), arguments='-Pn -p- -T4 -A -v')
    json_data = nm.analyse_nmap_xml_scan()
    analize = json_data["scan"]
    try:
        # Prompt about what the quary is all about
        prompt = "do a vulnerability analysis of {} and return a vulnerabilty report in json".format(
            analize)
        # A structure for the request
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1
        )
        response = completion.choices[0].text
    except KeyboardInterrupt:
        print("Bye")
        quit()
    return response


def p5(ip):
    nm.scan('{}'.format(
        ip), arguments='-Pn -sS -sU -T4 -A -PE -PP -PS80,443 -PA3389 -PU40125 -PY -g 53 --script=vuln')
    json_data = nm.analyse_nmap_xml_scan()
    analize = json_data["scan"]
    try:
        # Prompt about what the quary is all about
        prompt = "do a vulnerability analysis of {} and return a vulnerabilty report in json".format(
            analize)
        # A structure for the request
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1
        )
        response = completion.choices[0].text
    except KeyboardInterrupt:
        print("Bye")
        quit()
    return response


frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

label = customtkinter.CTkLabel(
    master=frame, text="GVA System")
label.pack(pady=12, padx=10)

entry1 = customtkinter.CTkEntry(master=frame, placeholder_text="API_KEY")
entry1.pack(pady=12, padx=10)
entry2 = customtkinter.CTkEntry(master=frame, placeholder_text="Target")
entry2.pack(pady=12, padx=10)
entry5 = customtkinter.CTkEntry(
    master=frame, placeholder_text="Attack (nmap/dns)")
entry5.pack(pady=12, padx=10)
entry4 = customtkinter.CTkEntry(master=frame, placeholder_text="Savefile.json")
entry4.pack(pady=12, padx=10)
entry3 = customtkinter.CTkEntry(
    master=frame, placeholder_text="Profile (Only Nmap)")
entry3.pack(pady=12, padx=10)
radiobutton_var = customtkinter.IntVar(value=1)
button = customtkinter.CTkButton(
    master=frame, text="Run", command=application)
button.pack(pady=12, padx=10)

root.mainloop()
