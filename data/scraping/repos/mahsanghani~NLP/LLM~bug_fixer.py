import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="##### Fix bugs in the below function\n \n### Buggy Python\nimport Random\na = random.randint(1,12)\nb = random.randint(1,12)\nfor i in range(10):\n    question = \"What is \"+a+\" x \"+b+\"? \"\n    answer = input(question)\n    if answer = a*b\n        print (Well done!)\n    else:\n        print(\"No.\")\n    \n### Fixed Python",
  temperature=0,
  max_tokens=182,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["###"]
)

# Prompt
# ##### Fix bugs in the below function
 
# ### Buggy Python
# import Random
# a = random.randint(1,12)
# b = random.randint(1,12)
# for i in range(10):
#     question = "What is "+a+" x "+b+"? "
#     answer = input(question)
#     if answer = a*b
#         print (Well done!)
#     else:
#         print("No.")
    
# ### Fixed Python
# Sample response
# import random
# a = random.randint(1,12)
# b = random.randint(1,12)
# for i in range(10):
#     question = "What is "+str(a)+" x "+str(b)+"? "
#     answer = int(input(question))
#     if answer == a*b:
#         print ("Well done!")
#     else:
#         print("No.")
