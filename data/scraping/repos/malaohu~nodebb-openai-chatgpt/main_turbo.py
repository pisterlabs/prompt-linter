import requests
import json
import openai
import uuid
import time
import re
import config as conf
openai.api_key = conf.API_KEY
tern = re.compile(r'<[^>]+>', re.S)


class nodebb_gpt():

    def __init__(self):
        self.api_route = {
            "notice": conf.NB_HOST+"/api/notifications?filter=mention",
            "replay": conf.NB_HOST+"/api/v3/topics/",
        }

    def get_unread(self, last_pid=0):
        notice_info = self.req_util("GET", self.api_route.get("notice"), None)
        if not notice_info or not isinstance(notice_info.get("notifications"), list):
            return None

        pid_list = [last_pid]
        for notic in notice_info.get("notifications", []):
            pid_list.append(notic.get("pid"))
            if notic and notic.get("pid", -1) > last_pid and notic.get("bodyLong") and notic.get("bodyLong").find("@ChatGPT") > -1:
                ask_str = notic.get("bodyLong")

                if ask_str.find("@ChatGPT") > -1 or ask_str.find("@CHATGPT")> -1 :
                    try:
                        result = self.gtp(self.format(ask_str))
                    except Exception as e:
                        print(e)
                        result = "哎呀, OpenAI接口可能出问题了，请稍后再试！我这就PM站长~ @malaohu "
                    print(self.send_post(notic.get("tid"), notic.get("pid"),
                          result, ask_str, notic.get("user").get("username")))
                    time.sleep(5)
            else:
                break
        return max(pid_list)


    def format(self, content):
        lines = content.split('\n')
        stack = [{"role": "system", "content": "我是JIKE机器人AI,基于OpenAI GPT-4语言模型, 服务于JIKE.info 社区。帖子中@我就可以啦！"}]
        users = {}

        for i in range(0,len(lines)):
            line = lines[i]
       
            count_quote = line.count('> ')
            if re.search('^.*说：', line):
                count_quote += 1
        
            if not users.get(str(count_quote)):
                users[str(count_quote)] = []

            users[str(count_quote)].append(line.lstrip('> '))
  
        for key in sorted(users.keys(), reverse=True):
            role = users.get(key)[0]
            if key == "0":
                cont = users.get(key)[0:]
            else:
                cont = users.get(key)[1:]

            if role == "ChatGPT 说：":
                role = "assistant"
            else:
                role = "user"
            stack.append({"role": role, "content": '\n'.join(cont).replace("@ChatGPT ", "")})
        return stack


    def gtp(self, message_list):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message_list)
        return completion.choices[0].message.content


    def send_post(self, tid, pid, content, quote_content="", username=""):
        '''
        帖子中发回复内容
        '''
        if quote_content:
            content = "\n" + username+" 说：\n" + re.sub(
                r'^', '> ',  quote_content, flags=re.M) + "\n\n" + content
           
        data = {"uuid": str(uuid.uuid1()), "tid": tid,
                "toPid": pid, "content": tern.sub("", content)}

        url = self.api_route.get("replay") + str(tid)

        return self.req_util("POST", url, {},  data)

    def req_util(self, method, url, params={}, data={}, retry=3):
        retry = retry - 1
        headers = {
            'Authorization': 'Bearer ' + conf.NB_TOKEN,
            'Content-Type': 'application/json'
        }
        try:
            response = requests.request(
                method, url, headers=headers, data=json.dumps(data))
            return json.loads(response.text)
        except Exception as e:
            print("ERROR: ", e)
            if retry > 0:
                return self.req_util(method, url, params=params, data=data, retry=retry)
            return None

    def doit(self):

        fileName = 'last_pid'
        while (1 == 1):
            ofile = open(fileName, "r")
            last_pid = ofile.read()
            ofile.close()

            print("last_pid", last_pid)
            _last_pid = self.get_unread(last_pid=int(last_pid))
            print("new_pid", _last_pid)
            if _last_pid and last_pid and _last_pid > int(last_pid):
                with open(fileName, 'w', encoding='utf-8') as file:
                    file.write(str(_last_pid))
            time.sleep(15)


if __name__ == '__main__':
    nodebb_gpt().doit()
