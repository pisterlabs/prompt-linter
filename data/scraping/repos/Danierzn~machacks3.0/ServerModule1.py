import anvil.email
import anvil.users
import anvil.server
import tables
from tables import app_tables
import cohere
from cohere.classify import Example
import anvil.mpl_util
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import metrics 


# This is a server module. It runs on the Anvil server,
# rather than in the user's browser.
#
# To allow anvil.server.call() to call functions here, we mark
# them with @anvil.server.callable.
# Here is an example - you can replace it with your own:
#
# @anvil.server.callable
# def say_hello(name):
#   print("Hello, " + name + "!")
#   return 42
#
co = cohere.Client('fZf3vVCtJkS69wYLEJWyr8WGRUupRJ4NnMSUwL0e') # This is your trial API key

@anvil.server.callable 
def rheaModel():
  """
  response = co.classify(  
    model='<model>',  
    inputs=inputs)
  """ 
  response = co.classify(
  model='48112639-5bee-4b80-8f70-1f5f60b645fe-ft',
  inputs=["I'm gonna kill myself.", "Today was really boring."]) # add more examples, here we should add parsed user data from social media accounts

  ys = [] 
  preds = [] 
  flag = 0 

  for line in response: 
    if line.prediction == "suicide": 
      ys.append(1) 
      flag = 1 
    else: 
      ys.append(0) 
    preds.append(line.confidence) 

  y = np.array(ys)  
  pred = np.array(preds) 

  fpr, tpr, thresholds = metrics.roc_curve(y, pred)

  auc = metrics.auc(fpr, tpr)   
  return auc, flag, fpr, tpr 

@anvil.server.callable
def get_submissions():
  return app_tables.submissions.search()

@anvil.server.callable
def search_submissions(query):
  result = app_tables.submissions.search()
  if query:
    result = [
      x for x in result
      if x['user'] in query
      or x['user'] in query
      or query in str(x['user_location'])
    ]
  return result


@anvil.server.callable
def make_plot(): 
    data = rheaModel() 
    plt.plot(data[2],data[3],label="data 1, auc="+str(data[0])) 
    return anvil.mpl_util.plot_image()
