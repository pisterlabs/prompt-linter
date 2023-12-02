import openai
import os
import json
import sys
import requests
import re
import traceback
import hashlib
import time
import re
import random
from difflib import SequenceMatcher
from collections import MutableMapping
from random import randint


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


crumbs = True


def flatten(dictionary, parent_key=False, separator='.'):
    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            if not value.items():
                items.append((new_key, None))
            else:
                items.extend(flatten(value, new_key, separator).items())
        elif isinstance(value, list):
            if len(value):
                for k, v in enumerate(value):
                    items.extend(
                        flatten({str(k): v}, new_key, separator).items())
            else:
                items.append((new_key, None))
        else:
            items.append((new_key, value))

    return dict(items)


def string_helper(json_dict):
    string_sequence = json_dict['sequence']
    string_sequence = string_sequence.replace("[", "", 1)
    string_sequence = string_sequence[::-1].replace("]", "", 1)[::-1]
    string_sequence = string_sequence.split('], [')
    string_sequence[0] = string_sequence[0].lstrip('[')
    string_sequence[-1] = string_sequence[-1].rstrip(']')

    return string_sequence

# helper function for similarity check


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# helper function to delete certain params
def delete_key(json_obj, key_to_delete):
    if isinstance(json_obj, dict):
        for key in list(json_obj.keys()):
            if key_to_delete in key:
                del json_obj[key]
            else:
                delete_key(json_obj[key], key_to_delete)
    elif isinstance(json_obj, list):
        for item in json_obj:
            delete_key(item, key_to_delete)


def myHash(text):
    try:
        m = hashlib.sha256()
        m.update(text.encode('utf-8'))
        return m.hexdigest()
    except:
        return "unable to hash"


def generate_GPT_log(swagger_example, gpt_content):
    if enable_gpt_logs == 'False':
        return

    key = myHash(swagger_example)
    if key == 'unable to hash':
        return
    dct = {}
    dct[key] = gpt_content
    with open('gptlogs/log_'+str(service)+'.txt', 'a') as fd:
        fd.write(json.dumps(dct))
        fd.write('\n')

# for body


def check_existing_hash(swagger_example):
    if enable_gpt_logs == 'False':
        return False, {}

    key = myHash(swagger_example)
    exists = False

    if not os.path.isfile('gptlogs/log_'+str(service)+'.txt'):
        return exists, {}

    else:
        with open('gptlogs/log_'+str(service)+'.txt', 'r') as fd:
            lines = fd.readlines()
            for line in lines:
                if line != '\n':
                    val_jsn = json.loads(line)
                    if str(key) in val_jsn.keys():
                        exists = True
                        return exists, val_jsn[str(key)]

    return exists, {}


def getBodyForUrl(urlToFind, previousResponse, GPTcontent, isFormData):
    exmple = ''
    try:
        for ms in microservices:
            host = ms['host']
            methodToRequestMap = ms['methodToRequestMap']
            for key in methodToRequestMap:
                if (key == "POST"):
                    requestList = methodToRequestMap[key]
                    for ele in requestList:
                        url = host + ele['url']
                        if (urlToFind == url):
                            if 'example' not in ele:
                                return "", exmple, isFormData
                            if 'contentType' in ele:
                                if ele['contentType'] == "FORM_DATA":
                                    isFormData = True
                            try:
                                exmple = ele['example']
                                exmple = json.loads(exmple)
                            except:
                                exmple = ''
                                pass
                            # check for GTP hash in log
                            exists, existing_val = check_existing_hash(
                                ele['example'])
                            if exists:
                                print("CACHE HIT")
                                return existing_val, exmple, isFormData

                            if not previousResponse:
                                print("GPT REQUEST: "+str(ele['example']))
                                if 'prompt' not in ele.keys():
                                    response = openai.ChatCompletion.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "system",
                                                "content": "You are an assistant that provides sample json data for HTTP POST requests. These are a sequence of HTTP requests so please use the same context in subsequent requests"},
                                            {"role": "user", "content": "using the same context provide one json data that follows the key value information in : {0}. Don't add any additional attributes and respond with only a json without additional information.".format(
                                                ele['example'])},
                                            {"role": "user", "content": "For values that could not be found from above context, use {} for the same. For dates use the format: yyyy-MM-dd'T'HH:mm:ss. Add +1 country code for phone numbers only if phone number is present in the json struture given. Return strong passwords for password field only if password is present in the json context given. Please provide full form values for all attributes in provided json structure".format(
                                                GPTcontent)}
                                        ]
                                    )
                                else:
                                    response = openai.ChatCompletion.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "system",
                                                "content": "You are an assistant that provides sample json data for HTTP POST requests. These are a sequence of HTTP requests so please use the same context in subsequent requests"},
                                            {"role": "user", "content": "using the same context provide one json data that follows the key value information in: {0}. Use {1} as reference to substitute for values in required places. Don't add any additional attributes and respond with only a json without additional information.".format(
                                                ele['example'], ele['prompt'])},
                                            {"role": "user", "content": "For values that could not be found from above context, use {} for the same. For dates use the format: yyyy-MM-dd'T'HH:mm:ss. Add +1 country code for phone numbers only if phone number is present in the json struture given. Return strong passwords for password field only if password is present in the json context given. Please provide full form values for all attributes in provided json structure".format(
                                                GPTcontent)}
                                        ]
                                    )
                                content = response['choices'][0]['message']['content']
                                content_json = content.split("{", 1)[1]
                                content_json = "{" + \
                                    content_json.rsplit("}", 1)[0] + "}"
                                print("GPT RESPONSE: " + str(content_json))
                                try:
                                    content_json = json.loads(content_json)
                                    generate_GPT_log(
                                        ele['example'], content_json)
                                except:
                                    content_json = {}
                                return content_json, exmple, isFormData
                            else:
                                print("GPT REQUEST: "+str(ele['example']))
                                if 'prompt' not in ele.keys():
                                    response = openai.ChatCompletion.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "system",
                                                "content": "You are a helpful assistant that provides sample json data for HTTP POST requests. These are a sequence of HTTP requests so please use the same context in subsequent requests"},
                                            {"role": "user", "content": "The previous POST request returned the json: {0}".format(
                                                previousResponse)},
                                            {"role": "user", "content": "using the same context and reusing the attribute values from the previous response, provide one json data that follows the json structure: {0}. Don't add any additional attributes and respond with only a json without additional information.".format(
                                                ele['example'])},
                                            {"role": "user", "content": "For values that could not be found from above context, use {} for the same. For dates use the format: yyyy-MM-dd'T'HH:mm:ss. Add +1 country code for phone numbers only if phone number is present in the json struture given. Return strong passwords for password field only if password is present in the json context given. Please provide full form values for all attributes in provided json structure".format(
                                                GPTcontent)}
                                        ]
                                    )
                                else:
                                    response = openai.ChatCompletion.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "system",
                                                "content": "You are a helpful assistant that provides sample json data for HTTP POST requests. These are a sequence of HTTP requests so please use the same context in subsequent requests"},
                                            {"role": "user", "content": "The previous POST request returned the json: {0} and some fields need to be populated with values in {1}".format(
                                                previousResponse, ele['prompt'])},
                                            {"role": "user", "content": "using the same context and reusing the attribute values from the previous response, provide one json data that follows the json structure: {0}. Don't add any additional attributes and respond with only a json without additional information.".format(
                                                ele['example'])},
                                            {"role": "user", "content": "For values that could not be found from above context, use {} for the same. For dates use the format: yyyy-MM-dd'T'HH:mm:ss. Add +1 country code for phone numbers only if phone number is present in the json struture given. Return strong passwords for password field only if password is present in the json context given. Please provide full form values for all attributes in provided json structure".format(
                                                GPTcontent)}
                                        ]
                                    )
                                content = response['choices'][0]['message']['content']
                                content_json = content.split("{", 1)[1]
                                content_json = "{" + \
                                    content_json.rsplit("}", 1)[0] + "}"
                                try:
                                    content_json = json.loads(content_json)
                                    generate_GPT_log(
                                        ele['example'], content_json)
                                except:
                                    content_json = {}
                                print("GPT RESPONSE: " + str(content_json))
                                return content_json, exmple, isFormData
    except Exception as e:
        print(traceback.format_exc())
    return '', exmple, isFormData


def getParamFromAlreadyGeneratedValues(allJsonKeyValues, param):
    paramSet = set()
    for i in allJsonKeyValues:
        for j in i:
            if len(paramSet) > 10:
                break
            param_new = param
            if param_new[-1] == 's':
                param_new = param_new[:-1]
            if param_new.lower() in j.lower() or j.lower() in param_new.lower() or similar(j.lower(), param_new.lower()) > 0.85:
                paramSet.add(i[j])
    return paramSet


def getParamFromChatGPT(postUrl, param, allJsonKeyValues):
    # check for existing params
    exists, existing_value = check_existing_hash(param)
    if exists:
        print("CACHE HIT")
        return existing_value

    print("GPT REQUEST: "+str(param))
    response2 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
                     "content": "You are working with HTTP POST request URLs. You will only provide a single word output."},
            {"role": "user", "content": "Can you give one valid path param value for the {} param in the POST URL {} without any other information. If you are unable to generate a value provide one unique identifier without any other information or text.Do not use abbreviations. Output should be a single word.".format(
                param, postUrl)}
        ]
    )

    content2 = response2['choices'][0]['message']['content']
    if content2.endswith("."):
        content2 = content2[:-1]

    if "\"" in content2 or "\'" in content2:
        match = re.search(
            r'"([^"]*)"', content2) or re.search(r"'([^']*)'", content2)
        content2 = match.group(1)

    data = {}
    data[param] = content2
    allJsonKeyValues.append(flatten(data))

    print("GPT RESPONSE: "+str(content2))
    # generate GPT log
    generate_GPT_log(param, content2)
    return content2


def processPostID(allJsonKeyValues, postUrl, postUrlIDVariation, microservices):
    if "{" not in postUrl:
        postUrlIDVariation.add(postUrl)
    else:
        for ms in microservices:
            host = ms['host']
            methodToRequestMap = ms['methodToRequestMap']
            for key in methodToRequestMap:
                if (key == "POST"):
                    requestList = methodToRequestMap[key]
                    for ele in requestList:
                        url = host + ele['url']
                        if (postUrl == url):
                            if 'pathParamExample' in ele.keys():
                                resp = ele['pathParamExample']
                                resp = json.loads(resp)
                                var = postUrl
                                for key in resp.keys():
                                    var = var.replace(
                                        "{"+key+"}", str(resp[key]))
                                postUrlIDVariation.add(var)

        allParams = re.findall('\{.*?\}', postUrl)
        for param in allParams:
            paramValues = getParamFromAlreadyGeneratedValues(
                allJsonKeyValues, param)
            if len(paramValues) == 0:
                paramFromChatGPT = getParamFromChatGPT(
                    postUrl, param, allJsonKeyValues)
                if (len(paramFromChatGPT) > 0):
                    stringVal = str(paramFromChatGPT)
                    tmp = postUrl
                    postUrl = postUrl.replace(param, stringVal)
                    postUrlIDVariation.add(postUrl)
                else:
                    tmp = postUrl
                    if "id" in param.lower():
                        postUrl = postUrl.replace(param, "1")
                        postUrlIDVariation.add(postUrl)
                    else:
                        postUrl = postUrl.replace(param, "")
                        postUrlIDVariation.add(postUrl)
            else:
                for p in paramValues:
                    tmp = postUrl
                    stringVal = str(p)
                    postUrl = postUrl.replace(param, stringVal)
                    postUrlIDVariation.add(postUrl)


def processGetRequests(allJsonKeyValues, getUrl, tmp, allIdFields, microservices):
    if "{" not in getUrl:
        tmp.add(getUrl)
    else:
        for ms in microservices:
            host = ms['host']
            methodToRequestMap = ms['methodToRequestMap']
            for key in methodToRequestMap:
                if (key == "GET"):
                    requestList = methodToRequestMap[key]
                    for ele in requestList:
                        url = host + ele['url']
                        if (getUrl == url):
                            if 'pathParamExample' in ele.keys():
                                resp = ele['pathParamExample']
                                resp = json.loads(resp)
                                var = getUrl
                                for key in resp.keys():
                                    var = var.replace(
                                        "{"+key+"}", str(resp[key]))
                                tmp.add(var)

        allParams = re.findall('{(.+?)}', getUrl)
        for param in allParams:
            paramValues = getParamFromAlreadyGeneratedValues(
                allJsonKeyValues, param)
            for p in paramValues:
                url = getUrl
                url = url.replace("{"+param+"}", str(p))
                tmp.add(url)
                paramOnly = param.replace("{", "").replace("}", "")
                if paramOnly not in allIdFields:
                    allIdFields[paramOnly] = paramValues
                else:
                    allIdFields[paramOnly].update(paramValues)


def replaceAdditionalParams(processedUrls):
    try:
        remove = []
        add = []
        for url in processedUrls:
            # check for GPT logs
            exists, gpt_param = check_existing_hash(url)
            if not exists:
                print("GPT REQUEST: "+str(url))
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system",
                         "content": "You are generating HTTP GET request url"},
                        {"role": "user", "content": "Replace the params between braces in the url {} with one realistic example value. Provide only the url as a response without any explanation.".format(
                            url)}
                    ]
                )
                gpt_param = response['choices'][0]['message']['content']
                print("GPT RESPONSE: "+str(gpt_param))
                # add into GPT logs
                generate_GPT_log(url, gpt_param)

            remove.append(url)
            add.append(gpt_param)
        for j in remove:
            processedUrls.remove(j)
        for j in add:
            processedUrls.append(j)

    except Exception as e:
        print(traceback.format_exc())

    return processedUrls[0] if processedUrls else []


def getPutValuesForJson(jsonStr, idJsonLoad):
    # check if data already exists from GPT
    exists, existing_val = check_existing_hash(jsonStr)
    if exists:
        print("CACHE HIT")
        return existing_val
    content_json = ''
    try:
        print("GPT REQUEST: "+str(jsonStr))
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that provides sample json data for HTTP PUT requests using the same context as the previous POST and GET requests."},
                {"role": "user", "content": "using the same context and reusing the id fields from the json {} provide one json data that follows the json structure: {}. Don't add any additional attributes and respond with only a json without additional information.".format(
                                            idJsonLoad, jsonStr)},
                {"role": "user", "content": "Using the same context, substitute existing attributes present in the json with United States related data for each field and don't add additional attributes to the json and return only the json response without any extra text."}
            ]
        )
        content = response['choices'][0]['message']['content']
        content_json = content.split("{", 1)[1]
        content_json = "{" + \
            content_json.rsplit("}", 1)[0] + "}"
        try:
            content_json = json.loads(content_json)
            print("GPT RESPONSE: "+str(content_json))
            generate_GPT_log(jsonStr, content_json)
        except:
            content_json = {}
        return content_json
    except Exception as e:
        print(traceback.format_exc())
    return content_json


def process_response_post(resp, url, body, GPTcontent, prevRespJson, allJsonKeyValues):
    try:
        try:
            resp_val = int(resp.text)
            if isinstance(resp_val, int):
                allJsonKeyValues.append({"id": resp_val})
                prevRespJson.append(str({"id": resp_val}))
                return
        except:
            pass

        GPTcontent.append(body)
        id_gen = url.split("/")[-1]
        id_gen = id_gen[:-1]
        resp_json = {}
        try:
            resp = resp.json()
        except:
            resp = ""
        if resp != "" and resp:
            for key in resp:
                if key == 'id':
                    resp_json[id_gen + key] = resp[key]
                else:
                    resp_json[key] = resp[key]

            flatten_resp = flatten(resp_json)
            delete_key(flatten_resp, '_links')
            allJsonKeyValues.append(flatten_resp)
            prevRespJson.append(str(flatten_resp))

    except Exception as e:
        print(traceback.format_exc())


def pre_run(microservices):
    allJsonKeyValues = []
    prevRespJson = []
    GPTcontent = []
    run(microservices, allJsonKeyValues, prevRespJson, GPTcontent)


def run(microservices, allJsonKeyValues, prevRespJson, GPTcontent):
    finalReqs = {}
    finalReqs['POST'] = {}
    finalReqs['GET'] = {}
    finalReqs['PUT'] = {}
    finalReqs['DELETE'] = {}
    finalReqs['PATCH'] = {}
    const_no = str(random.randint(-5, 6))
    const_no2 = '10001'
    const_str = "xyz"

    for ms in microservices:
        host = ms['host']
        methodToRequestMap = ms['methodToRequestMap']
        for key in methodToRequestMap:
            if (key == "POST"):
                requestList = methodToRequestMap[key]
                for ele in requestList:
                    url = host + ele['url']
                    finalReqs['POST'][url] = ""
            elif (key == "GET"):
                requestList = methodToRequestMap[key]
                for ele in requestList:
                    url = host + ele['url']
                    finalReqs['GET'][url] = ""
            elif (key == "PUT"):
                requestList = methodToRequestMap[key]
                for ele in requestList:
                    url = host + ele['url']
                    try:
                        exm = json.loads(ele['example'])
                        finalReqs['PUT'][url] = exm
                    except:
                        finalReqs['PUT'][url] = {}
            elif (key == "DELETE"):
                requestList = methodToRequestMap[key]
                for ele in requestList:
                    url = host + ele['url']
                    finalReqs['DELETE'][url] = ""
            elif (key == "PATCH"):
                requestList = methodToRequestMap[key]
                for ele in requestList:
                    url = host + ele['url']
                    try:
                        exm = json.loads(ele['example'])
                        finalReqs['PATCH'][url] = exm
                    except:
                        finalReqs['PATCH'][url] = {}

    # logically order the POST request using GPT
    print("START POST REQUEST")
    urls = ",".join(finalReqs['POST'].keys())
    if urls:
        urlList = urls.split(",")
        if len(urlList) > 2:
            print("GPT REQUEST: "+str(urls))
            response2 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are working with HTTP POST request URLs"},
                    {"role": "user", "content": "Can you logically order these POST URLs without any additional information as a comma separated line {}. Return only the urls as a comma separated string".format(
                        urls)}
                ]
            )

            content2 = response2['choices'][0]['message']['content']
            urlList = [x.strip() for x in content2.split(',')]

        print("GPT RESPONSE: " + str(urlList))

        for url in urlList:
            if url.endswith('.'):
                url = url[:-1]

            isFormData = False
            # Get body for POST request from GPT and also default body
            body_processed, body_def, isFormData = getBodyForUrl(
                url, prevRespJson, GPTcontent, isFormData)
            body_arr = []
            if body_processed:
                body_arr.append(body_processed)
            if body_def:
                body_arr.append(body_def)

            #  no body cases
            if len(body_arr) == 0:
                body = ""
                postUrlIDVariation = set()
                # get path parameter for POST request
                processPostID(allJsonKeyValues, url,
                              postUrlIDVariation, microservices)
                for postUrl in postUrlIDVariation:
                    if '{' not in postUrl:
                        print("POST URL : " + postUrl)
                        try:
                            resp = {}
                            headers = {}
                            headers['rest-tester'] = 'RESTGPT'
                            if isFormData:
                                headers['Content-type'] = 'application/x-www-form-urlencoded'
                                resp = requests.post(
                                    postUrl, json=body, headers=headers)
                            else:
                                resp = requests.post(
                                    postUrl, json=body, headers=headers)
                            print("INITIAL REQUEST: "+str(resp.status_code))
                            # process 200 response
                            if resp.status_code == 200 or resp.status_code == 201 or resp.status_code == 204:
                                process_response_post(
                                    resp, url, body, GPTcontent, prevRespJson, allJsonKeyValues)

                            # process 401 response. This can happen due to authentication
                            if resp.status_code == 401:
                                try:
                                    # Check if user has provided any token for authentication
                                    f = open('../input/headers/' +
                                             str(service)+'_header.json')
                                    headers = json.load(f)
                                except:
                                    pass

                                headers['rest-tester'] = 'RESTGPT'
                                if isFormData:
                                    headers['Content-type'] = 'application/x-www-form-urlencoded'
                                    resp = requests.post(
                                        postUrl, json=body, headers=headers)
                                else:
                                    resp = requests.post(
                                        postUrl, json=body, headers=headers)
                                print("PROCESS 401: " + str(resp.status_code))
                                if resp.status_code == 200 or resp.status_code == 201 or resp.status_code == 204:
                                    process_response_post(
                                        resp, url, body, GPTcontent, prevRespJson, allJsonKeyValues)

                        except Exception as e:
                            print(traceback.format_exc())

            # cases with body
            for body in body_arr:
                if body:
                    if isinstance(body, list):
                        for bdy_json in body:
                            if isinstance(bdy_json, str):
                                continue
                            else:
                                flatten_resp = flatten(bdy_json)
                                delete_key(flatten_resp, '_links')
                                allJsonKeyValues.append(flatten_resp)
                    else:
                        flatten_resp = flatten(body)
                        delete_key(flatten_resp, '_links')
                        allJsonKeyValues.append(flatten_resp)

                postUrlIDVariation = set()
                cov_url_no = ''
                cov_url_str = ''

                # Replace path parameters with random constants
                if '{' in url:
                    allParams = re.findall('{(.+?)}', url)
                    cov_url_no = url
                    cov_url_str = url
                    for param in allParams:
                        cov_url_no = cov_url_no.replace(
                            "{"+param+"}", const_no2)
                    postUrlIDVariation.add(cov_url_no)

                    for param in allParams:
                        cov_url_str = cov_url_str.replace(
                            "{"+param+"}", const_str)
                    postUrlIDVariation.add(cov_url_str)

                # Get path param from already existing values or GPT response
                processPostID(allJsonKeyValues, url,
                              postUrlIDVariation, microservices)

                for postUrl in postUrlIDVariation:
                    if "}" in postUrl:
                        postUrl = replaceAdditionalParams([postUrl])
                    if '{' not in postUrl:
                        print("POST URL : " + postUrl)
                        try:
                            resp = {}
                            headers = {}
                            headers['rest-tester'] = 'RESTGPT'
                            if isFormData:
                                headers['Content-type'] = 'application/x-www-form-urlencoded'
                                resp = requests.post(
                                    postUrl, json=body, headers=headers)
                            else:
                                resp = requests.post(
                                    postUrl, json=body, headers=headers)
                            print("INITIAL REQUEST: "+str(resp.status_code))
                            # process 200 response
                            if resp.status_code == 200 or resp.status_code == 201 or resp.status_code == 204:
                                process_response_post(
                                    resp, url, body, GPTcontent, prevRespJson, allJsonKeyValues)

                            # process 401 response. This can be due to authentication error
                            if resp.status_code == 401:
                                try:
                                    # Check if user has provided any token for authentication
                                    f = open('../input/headers/' +
                                             service+'_header.json')
                                    headers = json.load(f)
                                except:
                                    pass

                                headers['rest-tester'] = 'RESTGPT'
                                if isFormData:
                                    headers['Content-type'] = 'application/x-www-form-urlencoded'
                                    resp = requests.post(
                                        postUrl, json=body, headers=headers)
                                else:
                                    resp = requests.post(
                                        postUrl, json=body, headers=headers)
                                print("PROCESS 401: " + str(resp.status_code))
                                if resp.status_code == 200 or resp.status_code == 201 or resp.status_code == 204:
                                    process_response_post(
                                        resp, url, body, GPTcontent, prevRespJson, allJsonKeyValues)

                            # Process 400 response. This could be due to bad data, hence try to delete few attributes that might cause this
                            if resp.status_code == 400:
                                body_new = body
                                delete_key(body_new, "date")
                                if isFormData:
                                    headers['Content-type'] = 'application/x-www-form-urlencoded'
                                    resp = requests.post(
                                        postUrl, json=body_new, headers=headers)
                                else:
                                    resp = requests.post(
                                        postUrl, json=body_new, headers=headers)
                                print("PROCESS 400: "+str(resp.status_code))
                                if resp.status_code == 200 or resp.status_code == 201 or resp.status_code == 204:
                                    process_response_post(
                                        resp, url, body, GPTcontent, prevRespJson, allJsonKeyValues)

                                # handle cases where Id's are default and dates are missmatched
                                if resp.status_code == 400:
                                    body_new = body
                                    delete_key(body_new, "date")
                                    delete_key(body_new, "Time")
                                    post_checker = postUrl.split(
                                        "localhost:")[1]
                                    post_checker = post_checker.split("/")[1]
                                    keys_to_delete = []
                                    if isinstance(body_new, dict):
                                        for key in body_new.keys():
                                            if similar(key.lower(), "id") > 0.95:
                                                keys_to_delete.append(key)
                                            if similar(key.lower(), post_checker.lower()) > 0.60:
                                                keys_to_delete.append(key)

                                    for key in keys_to_delete:
                                        delete_key(body_new, key)

                                    if isFormData:
                                        headers['Content-type'] = 'application/x-www-form-urlencoded'
                                        resp = requests.post(
                                            postUrl, json=body_new, headers=headers)
                                    else:
                                        resp = requests.post(
                                            postUrl, json=body_new, headers=headers)
                                    print("PROCESS DEFAULTS: " +
                                          str(resp.status_code))
                                    if resp.status_code == 200 or resp.status_code == 201 or resp.status_code == 204:
                                        process_response_post(
                                            resp, url, body, GPTcontent, prevRespJson, allJsonKeyValues)

                        except Exception as e:
                            print(traceback.format_exc())

            postUrlIDVariation = []

    # start GET request
    allIdFields = {}
    print("START GET REQUESTS")
    getUrlsProcessed = []

    # logically order the get requests
    ordered_url = []
    for url in finalReqs['GET'].keys():
        if "{" not in url:
            ordered_url.append(url)

    for url in finalReqs['GET'].keys():
        if "{" in url:
            ordered_url.append(url)

    getUrlsProcessed = ordered_url

    for i in getUrlsProcessed:
        tmp = set()
        cov_url_no = ''
        cov_url_str = ''

        # replace path params with constants to increase negative scenarios
        if '{' in i:
            allParams = re.findall('{(.+?)}', i)
            cov_url_no = i
            cov_url_str = i
            for param in allParams:
                cov_url_no = cov_url_no.replace("{"+param+"}", const_no)
            tmp.add(cov_url_no)

            for param in allParams:
                cov_url_str = cov_url_str.replace("{"+param+"}", const_str)
            tmp.add(cov_url_str)

            random_int_neg = randint(-1*1000, 0)
            random_int_small = randint(1, 1000)
            random_int_big = randint(10**5, 10**10)
            random_int_deci = (randint(1, 5000))/100
            random_integers = [random_int_neg,
                               random_int_small, random_int_big, random_int_deci]
            for rnd in random_integers:
                const_url = i
                for param in allParams:
                    const_url = const_url.replace("{"+param+"}", str(rnd))
                tmp.add(const_url)

        tmp.add(i)
        # get path params
        processGetRequests(allJsonKeyValues, i,
                           tmp, allIdFields, microservices)
        try:
            for url in tmp:
                processed_url = replaceAdditionalParams([url])
                if '{' not in processed_url:
                    print("GET URL: " + processed_url)
                    headers = {'accept': '*/*'}
                    headers['rest-tester'] = 'RESTGPT'
                    resp = requests.get(processed_url, headers=headers)
                    print("INITIAL REQUEST: "+str(resp.status_code))
                    if resp.status_code == 200 or resp.status_code == 201 or resp.status_code == 204:
                        try:
                            inter_json = resp.json()
                            prevRespJson.append(str(inter_json))
                            limit = 0
                            if isinstance(inter_json, list):
                                for resp_jsn in inter_json:
                                    if resp_jsn is not None:
                                        if limit > 1:
                                            break
                                        flatten_resp = flatten(resp_jsn)
                                        delete_key(flatten_resp, '_links')
                                        size = len(flatten_resp)
                                        if size <= 100 and flatten_resp:
                                            allJsonKeyValues.append(
                                                flatten_resp)
                                            prevRespJson.append(
                                                str(flatten_resp))
                                        limit += 1
                            else:
                                flatten_resp = flatten(resp_jsn)
                                delete_key(flatten_resp, '_links')
                                size = len(flatten_resp)
                                if size <= 100 and flatten_resp:
                                    allJsonKeyValues.append(flatten_resp)
                                    prevRespJson.append(str(flatten_resp))
                        except:
                            pass

                    # process 401 response. This can be due to authentication error
                    if resp.status_code == 401:
                        try:
                            # Check if user has provided any token for authentication
                            f = open('../input/headers/' +
                                     str(service)+'_header.json')
                            headers = json.load(f)
                            headers['accept'] = '*/*'
                            
                        except:
                            pass
                        
                        headers['rest-tester'] = 'RESTGPT'
                        resp = requests.get(processed_url, headers=headers)
                        print("PROCESS 401: "+str(resp.status_code))
                        if resp.status_code == 200 or resp.status_code == 201 or resp.status_code == 204:
                            try:
                                inter_json = resp.json()
                                prevRespJson.append(str(inter_json))
                                limit = 0
                                if isinstance(inter_json, list):
                                    for resp_jsn in inter_json:
                                        if resp_jsn is not None:
                                            if limit > 1:
                                                break
                                            flatten_resp = flatten(resp_jsn)
                                            delete_key(flatten_resp, '_links')
                                            size = len(flatten_resp)
                                            if size <= 100 and flatten_resp:
                                                allJsonKeyValues.append(
                                                    flatten_resp)
                                                prevRespJson.append(
                                                    str(flatten_resp))
                                            limit += 1
                                else:
                                    flatten_resp = flatten(resp_jsn)
                                    delete_key(flatten_resp, '_links')
                                    size = len(flatten_resp)
                                    if size <= 100 and flatten_resp:
                                        allJsonKeyValues.append(flatten_resp)
                                        prevRespJson.append(str(flatten_resp))
                            except:
                                pass

        except Exception as e:
            print(traceback.format_exc())

    print("START PUT REQUESTS")
    finalProcessedPutReqs = {}
    for k in finalReqs['PUT'].keys():
        putUrlsProcessed = set()
        processGetRequests(allJsonKeyValues, k,
                           putUrlsProcessed, allIdFields, microservices)
        putUrlsProcessed = list(putUrlsProcessed)
        replaceAdditionalParams(putUrlsProcessed)
        for j in putUrlsProcessed:
            finalProcessedPutReqs[j] = finalReqs['PUT'][k]

    idJsonDump = json.dumps(allIdFields, default=set_default)
    idJsonLoad = json.loads(idJsonDump)
    print("final URL: "+str(finalProcessedPutReqs))
    for i in finalProcessedPutReqs:
        if '{' not in i:
            print("PUT URL: " + i)
            if finalProcessedPutReqs[i]:
                body_processed = getPutValuesForJson(
                    finalProcessedPutReqs[i], idJsonLoad)
            else:
                body_processed = {}
            body_arr = []
            body_arr.append(body_processed)
            body_arr.append(finalProcessedPutReqs[i])

            for body in body_arr:
                try:
                    headers = {'accept': '*/*'}
                    headers['rest-tester'] = 'RESTGPT'
                    if isFormData:
                        headers['Content-type'] = 'application/x-www-form-urlencoded'
                        resp = requests.post(
                            postUrl, json=body, headers=headers)
                    else:
                        resp = requests.put(i, json=body, headers=headers)
                    print("INITIAL REQUEST: "+str(resp.status_code))
                    if resp.status_code == 200 or resp.status_code == 201 or resp.status_code == 204:
                        flatten_resp = flatten(resp.json())
                        delete_key(flatten_resp, '_links')
                        allJsonKeyValues.append(flatten_resp)
                        prevRespJson.append(str(flatten_resp))

                    if resp.status_code == 401:
                        try:
                            f = open('../input/headers/' +
                                     str(service)+'_header.json')
                            headers = json.load(f)
                            headers['accept'] = '*/*'
                        except:
                            pass

                        headers['rest-tester'] = 'RESTGPT'
                        resp = requests.put(i, json=body, headers=headers)
                        print("PROCESS 401: " + str(resp.status_code))
                        if resp.status_code == 200 or resp.status_code == 201 or resp.status_code == 204:
                            flatten_resp = flatten(resp.json())
                            delete_key(flatten_resp, '_links')
                            allJsonKeyValues.append(flatten_resp)
                            prevRespJson.append(str(flatten_resp))

                except Exception as e:
                    print(traceback.format_exc())

    print("START PATCH REQUESTS")
    finalProcessedPatchReqs = {}
    for k in finalReqs['PATCH'].keys():
        putUrlsProcessed = set()
        processGetRequests(allJsonKeyValues, k,
                           putUrlsProcessed, allIdFields, microservices)
        putUrlsProcessed = list(putUrlsProcessed)
        replaceAdditionalParams(putUrlsProcessed)
        for j in putUrlsProcessed:
            finalProcessedPatchReqs[j] = finalReqs['PATCH'][k]

    idJsonDump = json.dumps(allIdFields, default=set_default)
    idJsonLoad = json.loads(idJsonDump)

    for i in finalProcessedPatchReqs:
        if '{' not in i:
            print("PATCH URL: " + i)
            body_processed = getPutValuesForJson(
                finalProcessedPatchReqs[i], idJsonLoad)
            body_arr = []
            body_arr.append(body_processed)
            body_arr.append(finalProcessedPatchReqs[i])

            for body in body_arr:
                try:
                    headers = {'accept': '*/*'}
                    headers['rest-tester'] = 'RESTGPT'
                    if isFormData:
                        headers['Content-type'] = 'application/x-www-form-urlencoded'
                        resp = requests.post(
                            postUrl, json=body, headers=headers)
                    else:
                        resp = requests.put(i, json=body, headers=headers)
                    print("INITIAL REQUEST: "+str(resp.status_code))
                    if resp.status_code == 200 or resp.status_code == 201 or resp.status_code == 204:
                        flatten_resp = flatten(resp.json())
                        delete_key(flatten_resp, '_links')
                        allJsonKeyValues.append(flatten_resp)
                        prevRespJson.append(str(flatten_resp))

                    if resp.status_code == 401:
                        try:
                            f = open('../input/headers/' +
                                     str(service)+'_header.json')
                            headers = json.load(f)
                            headers['accept'] = '*/*'
                        except:
                            pass

                        headers['rest-tester'] = 'RESTGPT'
                        resp = requests.patch(i, json=body, headers=headers)
                        print("PROCESS 401: " + str(resp.status_code))
                        if resp.status_code == 200 or resp.status_code == 201 or resp.status_code == 204:
                            flatten_resp = flatten(resp.json())
                            delete_key(flatten_resp, '_links')
                            allJsonKeyValues.append(flatten_resp)
                            prevRespJson.append(str(flatten_resp))

                except:
                    print(traceback.format_exc())

    print("START DELETE REQUESTS")
    deleteUrlsProcessed = set()
    for k in finalReqs['DELETE'].keys():
        processGetRequests(allJsonKeyValues, k,
                           deleteUrlsProcessed, allIdFields, microservices)

        deleteUrlsProcessed = list(deleteUrlsProcessed)
        replaceAdditionalParams(deleteUrlsProcessed)
        deleteUrlsProcessed = set(deleteUrlsProcessed)

    for i in deleteUrlsProcessed:
        if '{' not in i:
            print("DELETE URL: " + i)
            try:
                headers = {'accept': '*/*'}
                headers['rest-tester'] = 'RESTGPT'
                resp = requests.delete(i, json=body, headers=headers)
                print("INITIAL REQUEST: "+str(resp.status_code))
                if resp.status_code == 401:
                    try:
                        f = open('../input/headers/' +
                                 str(service)+'_header.json')
                        headers = json.load(f)
                        headers['accept'] = '*/*'
                    except:
                        pass
                        
                    headers['rest-tester'] = 'RESTGPT'
                    resp = requests.delete(i, json=body, headers=headers)
                    print("PROCESS 401: " + str(resp.status_code))

            except:
                print(traceback.format_exc())


if __name__ == "__main__":
    # Rest GPT tool.
    global service
    global enable_gpt_logs
    service = sys.argv[1]
    try:
        enable_gpt_logs = sys.argv[2]
    except:
        enable_gpt_logs = True

    try:
        runs = int(sys.argv[3])
    except:
        runs = 10
    
    # 70 seconds sleep time is provided to give time to start the service on which the tool is run.
    time.sleep(70)

    # load the openai key
    f = open('../input/constants.json')
    val = json.load(f)
    openai.api_key = val['apikey']
    openai.organization = os.getenv("OPENAI_ORGANIZATION")

    # input the unified swagger json
    f = open('../output/uiuc-api-tester-'+str(service)+'.json')
    microservices = json.load(f)

    # track 1 - Single service REST
    for i in range(runs):
        try:
            print("RUN STARTED FOR: " + str(i))
            pre_run(microservices)

        except Exception as e:
            print(traceback.format_exc())

    print("TRACK 1 DONE")

    # track 2 - Microservices
    try:
        # get the reverse topological orders
        dependency_file = open('../input/Sequence/'+str(service)+'.json')
        json_dict = json.load(dependency_file)
    except:
        print(traceback.format_exc())
        json_dict = {}

    if json_dict:
        for key in json_dict:
            if key == 'sequence':
                service_list = json_dict[key]
                for sequence_list in service_list:
                    for sequence_service in sequence_list:
                        index = 0
                        for swagger_service in microservices:
                            if swagger_service['microservice'] in sequence_service.strip() or sequence_service.strip() in swagger_service['microservice']:
                                try:
                                    print("RUN STARTED FOR SERVICE: " +
                                          str(sequence_service))
                                    pre_run([microservices[index]])
                                except:
                                    print(traceback.format_exc())
                            index += 1
