import openai

# create a chat completion


query = "what is the main goal of the paper?"

bad_results = [
 ('Below is part of the InstuctGPT paper. Could you read and summarize it to me?', '0.016393442'),
 ('What is funny about this image? Describe it panel by panel.', '0.009677419'),
 ('ada, babbage, and curie refer to models available via the OpenAI API '
  '[47].We believe that accurately predicting future capabilities is important '
  'for safety. Going forward we plan to reﬁne these methods and register '
  'performance predictions across various capabilities before large model '
  'training begins, and we hope this becomes a common goal in the ﬁeld.', '0.00952381')]


prompt = f"""
user query:
{query}

title of the paper: 
GPT-4 Technical Report

author:
OpenAI
 
year:
May 2023

excerpts:
[1]. {bad_results[0][0]}
[2]. {bad_results[1][0]}
[3]. {bad_results[2][0]}

---
given the excerpts from the paper above, answer the user query.
In your answer, make sure to cite the excerpts by its number wherever appropriate.
Note, however, that the excerpts may not be relevant to the user query.
"""
chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                               messages=[{"role": "user", "content": prompt}])

# print the chat completion
print(chat_completion.choices[0].message.content)


"""
The main goal of the paper titled "GPT-4 Technical Report" is not explicitly mentioned in the given excerpts. 
The excerpts primarily discuss the importance of accurately predicting future capabilities for safety, 
and the plan to refine methods and register performance predictions before training large models [3].
 However, without further information, it is not possible to determine the specific main goal of the paper.
"""


"""
That's pretty smart!
"""


# what if we change the excerpts to those that are more relevant?
good_results = [('As such, they have been the subject of substantial interest and progress in '
  'recent years [1–34].One of the main goals of developing such models is to '
  'improve their ability to understand and generate natural language text, '
  'particularly in more complex and nuanced scenarios. To test its '
  'capabilities in such scenarios, GPT-4 was evaluated on a variety of exams '
  'originally designed for humans.',
  8.97881326111729),
 ('Such models are an important area of study as they have the potential to be '
  'used in a wide range of applications, such as dialogue systems, text '
  'summarization, and machine translation. As such, they have been the subject '
  'of substantial interest and progress in recent years [1–34].One of the main '
  'goals of developing such models is to improve their ability to understand '
  'and generate natural language text, particularly in more complex and '
  'nuanced scenarios.',
  8.398702787987148),
 ('Predictions on the other ﬁve buckets performed almost as well, the main '
  'exception being GPT-4 underperforming our predictions on the easiest '
  'bucket. Certain capabilities remain hard to predict.',
  7.135548135175706)]


prompt = f"""
user query:
{query}

title of the paper: 
GPT-4 Technical Report

author:
OpenAI
 
year:
May 2023

excerpts:
[1]. {good_results[0][0]}
[2]. {good_results[1][0]}
[3]. {good_results[2][0]}

---
given the excerpts from the paper above, answer the user query.
In your answer, make sure to cite the excerpts by its number wherever appropriate.
Note, however, that the excerpts may not be relevant to the user query.
"""
chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                               messages=[{"role": "user", "content": prompt}])

print("####")
# print the chat completion
print(chat_completion.choices[0].message.content)

"""
The main goal of the paper titled "GPT-4 Technical Report" by OpenAI is to improve the understanding and generation of natural language text by developing models such as GPT-4 [1][2].
These models have the potential to be used in various applications like dialogue systems, text summarization, and machine translation [2].
The paper discusses evaluating the capabilities of GPT-4 in complex and nuanced scenarios by testing it on human-designed exams [1].
Additionally, the paper mentions the challenges of predicting certain capabilities of GPT-4 [3].
"""