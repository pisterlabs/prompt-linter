import cohere
from cohere.classify import Example
co = cohere.Client('yiOWD4KfXSiayGiim2MRmZRUvGsbdEFOY5QaCQ1Z') # This is your trial API key

# inputs = ["I love school", "This item was broken when it arrived", "School is so fun!", "This item broke after 3 weeks"]

examples=[
    Example("i felt anger when at the end of a telephone call", "anger"), 
    Example("i am feeling outraged it shows everywhere", "anger"), 
    Example("i feel my heart is tortured by what i have done", "anger"), 
    Example("i love that this is a place a series with no real heroes and i love that the way the couples in these books fall in love feels just as violent and crazy as the place that they call home", "anger"), 
    Example("i pay attention it deepens into a feeling of being invaded and helpless", "fear"), 
    Example("i cant walk into a shop anywhere where i do not feel uncomfortable", "fear"), 
    Example("i feel a little nervous i go to the gym", "fear"), 
    Example("ive been feeling afraid a lot lately", "fear"), 
    Example("i see momo feel shy momo hmmm gt me heyy momo", "fear"), 
    Example("im feel a little bit shy to talked to her for a second but manage myself because i saw from her eyes that theres something with this girl", "fear"), 
    Example("i just feel extremely comfortable with the group of people that i dont even need to hide myself", "joy"), 
    Example("i left with my bouquet of red and yellow tulips under my arm feeling slightly more optimistic than when i arrived", "joy"), 
    Example("i like to have the same breathless feeling as a reader eager to see what will happen next", "joy"), 
    Example("i feel amused looking at the little turtle who sneaked in with them", "joy"), 
    Example("im thankful because i feel somewhat energetic instead of the dead fish that i would become every time every chemo", "joy"), 
    Example("i feel so honored today and i want to share the emotion and my gratitude because i received a very complimentary email from someone who reads thought provoking perspectives", "joy"), 
    Example("i feel quite passionate about and that is how old should children be to undergo beauty treatments", "joy"), 
    Example("i will feel more lively and full of bounce", "joy"), 
    Example("i find myself in the odd position of feeling supportive of", "love"), 
    Example("i want each of you to feel my gentle embrace", "love"), 
    Example("i sometimes feel is carried in my heart just by loving my child so fiercely", "love"), 
    Example("i feel like breathing is as delicate as dried rose petals sometimes", "love"), 
    Example("im feeling rather rotten so im not very ambitious right now", "sadness"), 
    Example("i was feeling a little vain when i did this one", "sadness"), 
    Example("i stole a book from one of my all time favorite authors and now i feel like a rotten person", "sadness"), 
    Example("i feel like there is no way out being humiliated by asa a guy i was obssessed about who played an embarrassing joke on me getting caught by tabbys wife tabby is a lover i once had who was married and i blindly fell in love with him", "sadness"), 
    Example("ive tried bare minerals but it makes me feel like my face is dirty", "sadness"), 
    Example("i feel like i m going to struggle and fail and suffer and be really dumb", "sadness"), 
    Example("i think i wanted audiences to feel impressed inspired or entertained when i was on stage", "surprise"), 
    Example("i am feeling overwhelmed by trying to do it all that i think on the women before me", "surprise"), 
    Example("this was totally unexpected and I did not see it coming", "surprise"), 
    Example("the month does not feel as hard as I thought it would be", "surprise")
]


# response = co.classify(
#   model='large',
#   inputs=inputs,
#   examples=examples)
# print('The confidence levels of the labels are: {}'.format(response.classifications))


def sentiment_analysis(inputs):
  """
  Takes in a list of strings (sentences)
  Returns the classified emotion (string)
  """
  labels = ['anger', 'fear', 'sadness', 'joy', 'love', 'surprise']
  emotions = {'anger': 0, 'fear': 0, 'sadness': 0, 'joy': 0, 'love': 0, 'surprise': 0}
  response = co.classify(  
      model='large',  
      inputs=inputs,  
      examples=examples)
  # print('The confidence levels of the labels are: {}'.format(response.classifications))
  for i in range(len(inputs)):
    searching = True
    end = 29
    while searching:
      if (str(response.classifications[i])[end] == "\""):
          searching = False
      end += 1
    emotion = (str(response.classifications[i]))[28:end-1]
    emotions[emotion] += 1
  
  index_majority = max(emotions, key=emotions.get)

  return index_majority

# print(sentiment_analysis(inputs))