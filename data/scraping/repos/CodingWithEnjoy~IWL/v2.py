import openai
import json
import urllib.request
import os
import colorfan as cf

openai.api_key = "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"

msgs = [{"role": "system",
         "content": "You most only give user a json array whose information is as follows: member 'name' with the value of the real name of a library with the name of the user input, member 'dev' with the value of the developer of that library, member 'download' with the value of an array containing the cdn addresses of that library and member 'save' with the value of an array that The appropriate address to store each member ( scripts in .iwl/js and styles in .iwl/css) is in the 'download' member on the host and the 'useby' member contains an array where each member is an html tag to access the 'save' files; you most only give user a json array without any explanition"}]


def cmd():
    inp = input(cf.fgcolor(cf.green) + "iwl " + cf.uncolor())
    cdns = []
    name = "you want to install"
    print(cf.fgcolor(cf.blue) + "wait..." + cf.uncolor())
    for cdn in inp.split(' '):
        try:
            msgs.append({"role": "user", "content": cdn})
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=msgs
            )
            data = json.loads(completion.choices[0].message.content)[0]
            name = name + " " + data['name'] + " &"
            cdns.append(data)
        except:
            print(cf.fgcolor(cf.red) + "an error occurred" + cf.uncolor())
            cmd()
            break
    print(name[:-1] + "(y/n)")
    inp = input().lower()
    while inp != "y" and inp != "n":
        print("invalid input")
        inp = input().lower()
    if inp == "y":
        for cdn in cdns:
            print(cdn['name'] + " is on the way")
            print("made by " + cdn['dev'])
            print(cf.fgcolor(cf.blue) + "wait..." + cf.uncolor())
            j = []
            for i in range(len(cdn['download'])):
                try:
                    os.makedirs(os.path.dirname(cdn['save'][i]), exist_ok=True)
                    urllib.request.urlretrieve(
                        cdn['download'][i], cdn['save'][i])
                    print(cf.fgcolor(cf.green) + os.path.basename(
                        cdn['download'][i]) + cf.uncolor() + " downloaded successfully")
                except:
                    print(cf.fgcolor(cf.red) + os.path.basename(
                        cdn['download'][i]) + cf.uncolor() + " could not be downloaded")
                    print("try download it manually from " +
                          cdn['download'][i])
                    print("and save it in " + cdn['save'][i])
            if len(j) != 0:
                print(cf.fgcolor(cf.red) + os.path.basename(
                    "some files were not downloaded." + cf.uncolor()))
                print("to see them and solve their problems,")
                print("read the situations printed above.")
            else:
                print(cf.fgcolor(cf.green) +
                      "all files have been downloaded successfully" + cf.uncolor())

            print("these are html tags you most use to access this files :")

            for i in range(len(cdn['save'])):
                if i in j:
                    print(os.path.dirname(cf.fgcolor(cf.red) + cdn['save'][i]) + cf.uncolor() + " by " +
                          cf.fgcolor(cf.yellow) + cdn['useby'][i] + " (not downloaded)")
                else:
                    print(cf.fgcolor(cf.green) + os.path.dirname(
                        cdn['save'][i]) + cf.uncolor() + " by " + cf.fgcolor(cf.yellow) + cdn['useby'][i])
            cmd()
    else:
        cmd()
cmd()