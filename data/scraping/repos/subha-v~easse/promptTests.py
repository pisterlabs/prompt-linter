import cohere
import os
import re
from numpy import loadtxt

# Reading the txt files from datasets into python arrays
my_file = open('./new_datasets/test_1/TURK_Original.txt', 'r')
data = my_file.read()
splitting = data.split("\n")

# Calling the prompt iteratively on the dataset
co = cohere.Client('PsMBI2NlLAh8ZONu3j2x8UNZgEsj9lxZHgojVnn2')
simplified_array = []
#f = open('TURK_XLarge.txt', 'w')

for i in range (0, len(splitting)-1):
  response = co.generate(preset="simplifying-sentences-1-bl6ktw", prompt_vars={"sentence": f"{splitting[i]}"})
  simplified_text = response.generations[0].text
  with open('TURK_XLarge.txt', 'a') as f:
    f.write(simplified_text)
    #f.write('\n')
  
  f.close()
  
  print(simplified_text)
  simplified_array.append(simplified_text)

for i in range(0, len(simplified_array)):
  simplified_array[i] = simplified_array[i].replace('\n', '')

print(simplified_array)