# final.py

# @Author: Seyed Mohammad Reza Shahrestani
# @date: 22/04/2022


import huggingface
import openai
import convertor
import query as queryMaker


# Queries the database and sends the data along with the predicate to the language models
# by default it only uses the first label of the query parameters
# you can change this by adding the fifth parameter
# @params: String: name of the server, String: the query, 
# OPTIONAL: String: comes before the label, OPTIONAL: String: comes after the label
# INT: selects the index of the label user wants to use, by default this value is set to 0
# @return: [String: the sentence completed by openai], [String: the sentence completed by huggingface],
# [String: lables], [String: descriptions]
def get(server, query, before='', after='', answer=0):

    # Makes the query and gets the results
    mainQuery = queryMaker.getData(server,query)
    listOfVariables = mainQuery[0][1][answer]
    listOfDescriptions = mainQuery[1]

    # Initialising LM variables
    openaiList = []
    huggingfaceList = []

    print("Getting data from the Language Models, Please be patient...")

    # combines the label and the description of the query if it has 'description'
    # Otherwise it adds the 'before' and 'after' to the sentence
    # And send the sentances to the LMs to get the results
    for i in range(len(listOfVariables)):
        if len(listOfDescriptions) != 0 : sentance = listOfVariables[i] + ", " + listOfDescriptions[i]
        else : sentance = listOfVariables[i]
        openaiList.append(openai.complete(before + sentance + after))
        huggingfaceList.append(huggingface.complete(before + sentance + after))

    return openaiList, huggingfaceList, listOfVariables, listOfDescriptions


# Splits the predicate by sending the predicate to the convertor file
# @params: String: name of the server, String: the query, String: the predicate
# @return: [String: the sentence completed by openai], [String: the sentence completed by huggingface],
# [String: lables], [String: descriptions]
def magic(server, query, predicate):
  if predicate == '':
    get(server, query, predicate, 1)

  res = convertor.parse(predicate)
  if res == -1: return res
  elif res == 0: return get(server, query, predicate)
  else:
    return get(server, query, res[0], res[1])